import os
import json
import numpy as np
import scipy.io as sio
import argschema as ags

class IO_Schema(ags.ArgSchema):
    combos = ags.fields.List(ags.fields.String, 
                             cli_as_single_argument=False, 
                             dump_default=['_2', '_0_2', '_2_2'], 
                             metadata={'description' : "List of branching combinations."})
    
    vs_root = ags.fields.InputDir(metadata={'description' : "Directory with valid set .mat files"}, 
                                  dump_default=r'\\allen\programs\celltypes\workgroups\mousecelltypes\SarahWB\packages\tree_comparison\tree_comparison\data')

    out_file = ags.fields.OutputFile(metadata={'description' : "Path to output JSON file."}, 
                                     dump_default='valid_set.json')

def _convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist() 
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def hardcode_valid_sets(validSetDir, 
                        valid_set_dict_path,
                        combos):
    
    #assemble valid sets dict
    validSet_dict = {}  
    for combo in combos: 
        filepath = os.path.join(validSetDir, combo+'.mat')
        if os.path.isfile(filepath):
            validSet = sio.loadmat(filepath)
            pOT1 = validSet['pOT1'][:,0]
            vs = validSet['vs'][0]

            validSet_dict[combo] = {'pOT1' : pOT1,
                                    'vs' : vs}

    # Write dict to JSON file
    with open(valid_set_dict_path, 'w') as f:
        json.dump(validSet_dict, f, default=_convert_numpy)


def main(args):

    hardcode_valid_sets(validSetDir=args['vs_root'],
                        valid_set_dict_path=args['out_file'],
                        combos=args['combos'])


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)
