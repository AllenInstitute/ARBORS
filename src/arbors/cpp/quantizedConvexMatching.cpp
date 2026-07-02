#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <utility>
#include <boost/multi_array.hpp>

namespace py = pybind11;

double computeMaxSim(std::vector<double> lengths1, std::vector<double> orientations1, std::vector<double> areas1, std::vector<int> pillars1,
                     std::vector<double> lengths2, std::vector<double> orientations2, std::vector<double> areas2, std::vector<int> pillars2) {

    int m = lengths1.size() + 1;
    int n = lengths2.size() + 1;

    double maxSimFinal = 0.0;
    double tmp, minArea, thirdEdgeLength;
    int maximumSegmentLength1 = 0, maximumSegmentLength2 = 0;
    int i, j, k, dim;

    for (i=0;i<m;i++)
		maximumSegmentLength1 = std::max(i-pillars1[i],maximumSegmentLength1);
    for (j=0;j<n;j++)
        maximumSegmentLength2 = ((j-pillars2[j])>maximumSegmentLength2)?(j-pillars2[j]):maximumSegmentLength2;

    double cumSumLen1[m], cumSumLen2[n];
    cumSumLen1[0]=0;
    cumSumLen2[0]=0;
    for (i=1;i<m;i++)
        if (i==pillars1[i]+1)
                cumSumLen1[i] = lengths1[i-1];
        else
                cumSumLen1[i] = cumSumLen1[i-1]+lengths1[i-1];
    for (j=1;j<n;j++)
        if (j==pillars2[j]+1)
                cumSumLen2[j] = lengths2[j-1];
        else
                cumSumLen2[j] = cumSumLen2[j-1]+lengths2[j-1];

	boost::multi_array< double, 2 > maxSimMatrix( boost::extents[m][n] );
	boost::multi_array< double, 3 > simMatrixPillar1( boost::extents[m][n][maximumSegmentLength2] );
    boost::multi_array< double, 3 > simMatrixPillar2( boost::extents[m][maximumSegmentLength1][n] );

	if(m==0 || n==0)
	{
		maxSimFinal = (double)0;
		return maxSimFinal;
	}

        for (j=1;j<n;j++)
	{
		simMatrixPillar2[0][0][j]=0;
		for (k=0;k<j-pillars2[j];k++)
			simMatrixPillar1[0][j][k]=0;
	}
	for (i=0;i<m;i++)
	{
		simMatrixPillar1[i][0][0]=0;
		for (k=0;k<i-pillars1[i];k++)
			simMatrixPillar2[i][k][0]=0;
	}
        for (i=1;i<m;i++)
		for (j=1;j<n;j++)
		{
			minArea = (areas1[i-1]<areas2[j-1])?areas1[i-1]:areas2[j-1];
			for (k=0;k<j-pillars2[j];k++)
			{
                thirdEdgeLength = 0;
                for (dim=0;dim<3;dim++)
                {
					if (k > 0)
                        tmp = orientations1[3*i+dim]-orientations2[3*j+dim]+orientations2[3*(pillars2[j]+k)+dim];
					else
						tmp = orientations1[3*i+dim]-orientations2[3*j+dim];
                        thirdEdgeLength = thirdEdgeLength + tmp * tmp;
                }
                thirdEdgeLength = sqrt(thirdEdgeLength);
				if (k > 0)
                    simMatrixPillar1[i][j][k]=(cumSumLen1[i]+cumSumLen2[j]-cumSumLen2[pillars2[j]+k] - thirdEdgeLength)*minArea;
				else
                    simMatrixPillar1[i][j][k]=(cumSumLen1[i]+cumSumLen2[j] - thirdEdgeLength)*minArea;
			}
            for (k=0;k<i-pillars1[i];k++)
            {
                thirdEdgeLength = 0;
                for (dim=0;dim<3;dim++)
                {
                    if (k > 0)
                        tmp = orientations1[3*i+dim]-orientations2[3*j+dim]-orientations1[3*(pillars1[i]+k)+dim];
                    else
                        tmp = orientations1[3*i+dim]-orientations2[3*j+dim];
                        thirdEdgeLength = thirdEdgeLength + tmp * tmp;
                }
                thirdEdgeLength = sqrt(thirdEdgeLength);
                if (k > 0)
                    simMatrixPillar2[i][k][j]=(cumSumLen2[j]+cumSumLen1[i]-cumSumLen1[pillars1[i]+k] - thirdEdgeLength)*minArea;
                else
                    simMatrixPillar2[i][k][j]=(cumSumLen2[j]+cumSumLen1[i] - thirdEdgeLength)*minArea;
            }
		}

        for (i=0;i<m;i++)
		    maxSimMatrix[i][0]=0;
        for (j=1;j<n;j++)
		    maxSimMatrix[0][j]=0;
        for (i=1;i<m;i++)
            for (j=1;j<n;j++)
            {
                maxSimMatrix[i][j] = (maxSimMatrix[i-1][j]>maxSimMatrix[i][j-1])?maxSimMatrix[i-1][j]:maxSimMatrix[i][j-1];
			    for (k=0;k<i-pillars1[i];k++)
			    {
				    tmp = maxSimMatrix[pillars1[i]+k][pillars2[j]]+simMatrixPillar2[i][k][j];
				    maxSimMatrix[i][j] = (maxSimMatrix[i][j]>tmp)?maxSimMatrix[i][j]:tmp;
			    }
                for (k=0;k<j-pillars2[j];k++)
                {
                    tmp = maxSimMatrix[pillars1[i]][pillars2[j]+k]+simMatrixPillar1[i][j][k];
                    maxSimMatrix[i][j] = (maxSimMatrix[i][j]>tmp)?maxSimMatrix[i][j]:tmp;
                }
            }

	maxSimFinal = maxSimMatrix[m-1][n-1];
    return maxSimFinal;
}


PYBIND11_MODULE(quantized_convex_matching, m) {
    m.doc() = "Quantized convex matching function";

    m.def("quantized_convex_matching", &computeMaxSim, "Calculate maximum similarity between two tree edges");
}
