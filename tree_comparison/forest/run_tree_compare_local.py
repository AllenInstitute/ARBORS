import os
from math import pi
import argschema as ags
import itertools
from importlib.resources import files
from neuron_morphology.constants import AXON, BASAL_DENDRITE, APICAL_DENDRITE
import subprocess
import concurrent.futures
import multiprocessing
import numexpr as ne
import pickle
import ntpath

class IO_Schema(ags.ArgSchema):
    input_swc_file_1 = ags.fields.InputFile(dump_default=None, metadata={'description' : "1st swc file to load"}, allow_none=True)
    input_swc_file_2 = ags.fields.InputFile(dump_default=None, metadata={'description' : "2nd swc file to load"}, allow_none=True)
    input_swc_dir = ags.fields.InputDir(dump_default=None, metadata={'description' : "directory of swc files to make nXn comparison matrix for"}, allow_none=True)
    input_ref_dir = ags.fields.InputDir(dump_default=None, metadata={'description' : "directory of swc files to use as reference cells to compare all swcs in input_swc_dir -OR- input_swc_file_1 to"}, allow_none=True)

    output_dir = ags.fields.OutputDir(metadata={'description' : "dir to output jsons"}, dump_default=None, allow_none=True)
    
    similarity_function = ags.fields.String(metadata={'description' : "Similarity function to use. Options: 'length or convex'"}, dump_default='length')
    max_depth = ags.fields.Int(metadata={'description' : "Max depth to use for algorithm"}, dump_default=1)
    number_of_rotations = ags.fields.Int(metadata={'description' : "Number of evenly sampled Tree2 rotations (around y axis) for comparison"}, dump_default=1)

    valid_set_dir = ags.fields.InputDir(metadata={'description' : "Directory with valid set files"}, dump_default=str(files('tree_comparison') / "data"))
    valid_set_dict = ags.fields.InputFile(metadata={'description' : "JSON file with hardcoded valid sets"}, dump_default=os.path.join(str(files('tree_comparison') / "data"), 'validSet_mouse_inh_viz_ctx.json'))
   
    partition_length = ags.fields.Float(metadata={'description' : "Partition length for downsampling tree branch"}, dump_default=1/2000)
    angle_threshold = ags.fields.Float(metadata={'description' : "Angle threshold for downsampling tree branch"}, dump_default=pi/9)
    segment_threshold = ags.fields.Float(metadata={'description' : "Segment threshold for downsampling tree branch"}, dump_default=1/200)

    downsample_spacing = ags.fields.Float(metadata={'description' : "Segment threshold for downsampling tree branch"}, dump_default=None, allow_none=True)

def run_command(command):
    try:
        #run the command
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        #print any errors
        print(f"Command failed: {command}")
        print(f"Error:\n{e.stderr}")


def main(args):

    #set num cores/threads to be max machine has
    num_cores = multiprocessing.cpu_count() # Detect the number of available CPU cores
    if num_cores == 32: num_cores = 8 #reduce for ashwinb-ux1 memory error. 
    os.environ['NUMEXPR_MAX_THREADS'] = str(num_cores) # Set NUMEXPR_MAX_THREADS to the number of cores
    ne.set_num_threads(num_cores) # Initialize numexpr to use this setting
    print(f"NumExpr is using {ne.detect_number_of_cores()} cores and {ne.get_num_threads()} threads.")


    #check input file paths exist 
    input_file_1 = args['input_swc_file_1']
    input_file_2 = args['input_swc_file_2']
    input_swc_dir = args['input_swc_dir']
    input_ref_dir = args['input_ref_dir']

    #load a set of the completed json files that we don't need to submit jobs for
    pickle_file = '/allen/programs/celltypes/workgroups/mousecelltypes/SarahWB/datasets/tree_comparison_sfn/mouse_viz_ctx/20240829/data/tree_comparison/completed_jsons_ashwinb-ux1.pkl'
    with open(pickle_file, "rb") as file: completed_jsons = pickle.load(file)

    if all([input_obj is None for input_obj in [input_file_1, input_file_2, input_swc_dir]]):
        msg = "No input swc files provided for comparison, no input directory provided. Nothing to do"
        raise ValueError(msg)

    if all([input_obj is not None for input_obj in [input_file_1, input_file_2, input_swc_dir]]):
        msg = "Input swc files provided and input directory provided. Either provide input swc paths OR provide directory with swc files, not both."
        raise ValueError(msg)
    
    #get number of rotations for tree2 (only relevant when using convex simfunc)
    if args['similarity_function'] == 'length': num_rotations == 1 #rotation won't change sim score when using length function
    else: num_rotations = args['number_of_rotations']
    if num_rotations < 1: num_rotations = 1
    orientations = [i * (360 / num_rotations) for i in range(num_rotations)]

    compartments = [AXON, BASAL_DENDRITE]

    ########## Step 1: get the pairs of cells to compare to eachother. ##########

    # OPTION 1: compare to reference dir of swcs 
    if input_ref_dir is not None:

        ref_files = [os.path.join(input_ref_dir,f) for f in os.listdir(input_ref_dir) if f.endswith(".swc")]
        if len(ref_files) < 1:
            msg = "At least one reference swc file required in the input directory."
            raise ValueError(msg)
        
        # OPTION 1a: compare each swc in the input swc dir to all the reference swcs
        if input_swc_dir is not None:

            swc_files = [os.path.join(input_swc_dir,f) for f in os.listdir(input_swc_dir) if f.endswith(".swc")]
            if len(swc_files) < 1:
                msg = "At least one swc file required in the input directory."
                raise ValueError(msg)
            
            pairs = list(itertools.product(swc_files, ref_files))


        # OPTION 1b: compare a single swc to all the reference swcs
        elif input_file_1 is not None:

            if not all([os.path.exists(p) for p in [input_file_1]]):
                msg = "Input SWC Paths Do Not Exist. Please check the path provided: \n{} \n{}".format(input_file_1)
                raise ValueError(msg)
            
            pairs = list(itertools.product([input_file_1], ref_files))
            
        else: 
            msg = "A directory of swcs or a single swc must be given to compare the the reference dir files."
            raise ValueError(msg)

    # OPTION 2: compare cells within swc_dir to each other (no reference dir)
    elif input_swc_dir is not None:

        swc_files = [os.path.join(input_swc_dir,f) for f in os.listdir(input_swc_dir) if f.endswith(".swc")]
        if len(swc_files) < 2:
            msg = "At least two swc files required in the input directory. Only {} found".format(len(input_swc_dir))
            raise ValueError(msg)

        pairs = list(itertools.combinations(swc_files, 2))

    # OPTION 3: compare file1 to file2
    else:

        if not all([os.path.exists(p) for p in [input_file_1, input_file_2]]):
            msg = "Input SWC Paths Do Not Exist. Please check the paths provided: \n{} \n{}".format(input_file_1, input_file_2)
            raise ValueError(msg)
        
        pairs = [(input_file_1, input_file_2)]


    ########## Step 2: submit job files for all comparisons. ##########

    # #compare each pair of swcs with all specified orientations. 
    # job_dir = os.path.join(args['output_dir'], "JobFiles")
    # os.makedirs(job_dir, exist_ok=True)

    # pool_rotations = args['pool_rotations'] #True == run all rotations within the same job file. False == write a job file for each comparison. 

    print('Generating commands...')
    tree_comp_commands = []
    for pair in pairs:
        swc_1_path = pair[0]
        swc_2_path = pair[1]

        json_root = '{}_{}'.format(ntpath.basename(swc_1_path).rsplit('.',1)[0], ntpath.basename(swc_2_path).rsplit('.',1)[0])
        json_files = {j for j in completed_jsons if j.startswith(json_root)}
        if len(json_files) < num_rotations * len(compartments):
            #run this comparison command --> all the rots/comps haven't completed 

            # what you want to run locally
            tree_comp_command_kwargs = {'swc_1_path': swc_1_path,
                                        'swc_2_path': swc_2_path,
                                        'compartments': compartments,
                                        'output_dir': args['output_dir'],
                                        'similarity_function': args['similarity_function'],
                                        'max_depth' : args['max_depth'],
                                        'orientation' : orientations, 
                                        'valid_set_dir' : args['valid_set_dir'],
                                        'valid_set_dict' : args['valid_set_dict'],
                                        'partition_length' : args['partition_length'],
                                        'angle_threshold' : args['angle_threshold'],
                                        'segment_threshold' : args['segment_threshold'],
                                        'downsample_spacing' : args['downsample_spacing']
                                        }

            # Filter out None values and format the arguments
            tree_comp_command_kwargs = " ".join(["--{} {}".format(k, val) if not isinstance(val, list) else
                                        "--{} ".format(k) + " ".join(["{}".format(elem) for elem in val])
                                        for k, val in tree_comp_command_kwargs.items() if val is not None])
            
            tree_comparison_command = "tree-comparison {}".format(tree_comp_command_kwargs)
            tree_comp_commands.append(tree_comparison_command)

    # Run commands in parallel
    print('Running commands in parallel...')
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(run_command, tree_comp_commands)

    print('Done!')

def console_script():
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)
