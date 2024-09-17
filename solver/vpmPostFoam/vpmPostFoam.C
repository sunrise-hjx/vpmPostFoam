/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2015 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.
\*---------------------------------------------------------------------------*/

#include <random>
#include <iostream>

#include "fvCFD.H"
#include "upwind.H"
#include "OFstream.H"
#include "syncTools.H"
#include "vectorList.H"
#include "tensorList.H"
#include "pyrMatcher.H"
#include "tetMatcher.H"
#include "hexMatcher.H"
#include "DynamicList.H"
#include "prismMatcher.H"
#include "simpleControl.H"
#include "scalarMatrices.H"
#include "emptyPolyPatch.H"
#include "slicedSurfaceFields.H"
#include "volPointInterpolation.H"

#include "argList.H"
#include "timeSelector.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void reverseArray(label arr[], label size) 
{
    for (int i = 0; i < size / 2; i++) 
    {
        label temp = arr[i];
        arr[i] = arr[size - i - 1];
        arr[size - i - 1] = temp;
    }
}

int main(int argc, char *argv[])
{   
    timeSelector::addOptions(true, true);
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "vpmPostDict.H"

    label noe=mesh.nCells();
    label numVertex = 0;
    label numElement = 0;
    labelListList mixFaceLabels(noe);
    labelListList mixVertLabels(noe);
    labelList mixMeshType(noe);

    labelListList cellPointsType;

    if(Dimension==2)
    {
        #include "cellMatch2d.H"
        #include "facePoints.H"

        numVertex = mesh.nPoints() / 2;

        forAll(mesh.C(),i)
        {
            numElement += mesh.cells()[i].size() - 2;
        }
        cellPointsType.setSize(numElement);

        label elementCount = -1;
        forAll(mesh.C(), i)
        {
            if(mixMeshType[i]==0)
            {
                for (label fi = 0; fi < 4; fi++)
                {
                    elementCount++;
                    cellPointsType[elementCount].append(5);
                    label faceI = mixFaceLabels[i][fi];
                    forAll(facePoints[faceI],pi)
                    {
                        cellPointsType[elementCount].append(facePoints[faceI][pi]);
                    }
                    cellPointsType[elementCount].append(i + numVertex);
                }
                
            }
            else
            {
                for (label fi = 0; fi < 3; fi++)
                {
                    elementCount++;
                    cellPointsType[elementCount].append(5);
                    label faceI = mixFaceLabels[i][fi];
                    forAll(facePoints[faceI],pi)
                    {
                        cellPointsType[elementCount].append(facePoints[faceI][pi]);
                    }
                    cellPointsType[elementCount].append(i + numVertex);
                }
                
            }
        }
    }
    else if(Dimension==3)
    {
        #include "cellMatch3d.H"
        numVertex = mesh.nPoints();

        forAll(mesh.C(),i)
        {
            numElement += mesh.cells()[i].size();
        }
        cellPointsType.setSize(numElement);

        label elementCount = -1;
        forAll(mesh.C(), i)
        {
            for (label fi = 0; fi < mesh.cells()[i].size(); fi++)
            {
                elementCount++;
                label faceI = mesh.cells()[i][fi];

                if(mesh.faces()[faceI].size()==3)
                {
                    cellPointsType[elementCount].append(10);

                    label pointIndex[3];
                    forAll(mesh.faces()[faceI],pi)
                    {
                        pointIndex[pi] = mesh.faces()[faceI][pi];
                    }

                    if(mesh.owner()[faceI] == i)
                    {
                        reverseArray(pointIndex, 3);
                    }

                    forAll(mesh.faces()[faceI],pi)
                    {
                        cellPointsType[elementCount].append(pointIndex[pi]);
                    }
                    cellPointsType[elementCount].append(i + numVertex);
                }
                else if(mesh.faces()[faceI].size()==4)
                {
                    cellPointsType[elementCount].append(14);

                    label pointIndex[4];
                    forAll(mesh.faces()[faceI],pi)
                    {
                        pointIndex[pi] = mesh.faces()[faceI][pi];
                    }

                    if(mesh.owner()[faceI] == i)
                    {
                        reverseArray(pointIndex, 4);
                    }

                    forAll(mesh.faces()[faceI],pi)
                    {
                        cellPointsType[elementCount].append(pointIndex[pi]);
                    }
                    cellPointsType[elementCount].append(i + numVertex);
                }
            }
        }
    }
    else
    {
        WarningInFunction << "The number of dimensions is wrong, it can only be 2D or 3D.";
        exit(1);
    }

    #include "boundarySearch.H"

    #include "pointPtNum.H"

    #include "cellNeighOwner.H"

    #include "leastSqureSVD.H"

    #include "fields.H"

    instantList timeDirs = timeSelector::select
    (
        runTime.findTimes(runTime.path(), runTime.constant()),
        args
    );

    if (timeDirs.empty())
    {
        WarningInFunction << "No times selected";
        exit(1);
    }

    forAll(timeDirs, timei)
    {
        runTime.setTime(timeDirs[timei], timei);
        Foam::Info << "Time = " << runTime.timeName() << Foam::nl << Foam::nl;
        #include "createFields.H"
        volTensorField gradUc(fvc::grad(U));
        UpGrad = tensor::zero;
        if(Q_Output || Omega_Output || U_Output)
        {
            forAll(gradUc, cellI)
            {
                gradUc[cellI] = tensor(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                scalarRectangularMatrix &aSVDc = leastSquareDiffSVDc[cellI];
                const labelList& cPoints = mesh.cellPoints()[cellI];
                const labelList& cCells =  mesh.cellCells()[cellI];
                forAll(cCells, cCellI)
                {
                    vector xx = Ud[cCells[cCellI]] - Ud[cellI];
                    vector gd = mesh.C()[cCells[cCellI]] - mesh.C()[cellI];
                    doubleScalar dd[9]={gd[0],gd[1],gd[2],gd[0]*gd[0],gd[1]*gd[1],gd[2]*gd[2],gd[0]*gd[1],gd[0]*gd[2],gd[1]*gd[2]};
                    vector ddx, ddy, ddz;
                    ddx = ddy = ddz = vector::zero;
                    for (label j = 0; j < 9; j++)
                    {
                        ddx += aSVDc[0][j] * dd[j] * xx;
                        ddy += aSVDc[1][j] * dd[j] * xx;
                        ddz += aSVDc[2][j] * dd[j] * xx;
                    }
                    gradUc[cellI] += tensor(ddx, ddy, ddz);
                }

                forAll(cPoints,cPointI)
                {
                    vector xx = Up[cPoints[cPointI]] - Ud[cellI];
                    vector gd=mesh.points()[cPoints[cPointI]]-mesh.C()[cellI];
                    doubleScalar dd[9]={gd[0],gd[1],gd[2],gd[0]*gd[0],gd[1]*gd[1],gd[2]*gd[2],gd[0]*gd[1],gd[0]*gd[2],gd[1]*gd[2]};
                    vector ddx, ddy, ddz;
                    ddx = ddy = ddz = vector::zero;
                    for (label j = 0; j < 9; j++)
                    {
                        ddx += aSVDc[0][j] * dd[j] * xx;
                        ddy += aSVDc[1][j] * dd[j] * xx;
                        ddz += aSVDc[2][j] * dd[j] * xx;
                    }
                    gradUc[cellI] += tensor(ddx, ddy, ddz);
                }
            }
            gradUc.correctBoundaryConditions();
            forAll(Up, pointI)
            {
                if(isPatchPoint_[pointI]) continue;

                scalarRectangularMatrix& aSVD = leastSquareDiffSVD[pointI];
                const labelList& pCells=mesh.pointCells()[pointI];
                const labelList& pPoints=mesh.pointPoints()[pointI];
                forAll(pCells,pCellI)
                {
                    vector xx = Ud[pCells[pCellI]] - Up[pointI];
                    vector gd = mesh.C()[pCells[pCellI]] - mesh.points()[pointI];
                    doubleScalar dd[9]={gd[0],gd[1],gd[2],gd[0]*gd[0],gd[1]*gd[1],gd[2]*gd[2],gd[0]*gd[1],gd[0]*gd[2],gd[1]*gd[2]};
                    vector ddx, ddy, ddz;
                    ddx = ddy = ddz = vector::zero;
                    for (label j = 0; j < 9; j++)
                    {
                        ddx += aSVD[0][j] * dd[j] * xx;
                        ddy += aSVD[1][j] * dd[j] * xx;
                        ddz += aSVD[2][j] * dd[j] * xx;
                    }
                    UpGrad[pointI] += tensor(ddx, ddy, ddz);
                }
                forAll(pPoints,pPointI)
                {
                    vector xx = Up[pPoints[pPointI]] - Up[pointI];
                    vector gd=mesh.points()[pPoints[pPointI]]-mesh.points()[pointI];
                    doubleScalar dd[9]={gd[0],gd[1],gd[2],gd[0]*gd[0],gd[1]*gd[1],gd[2]*gd[2],gd[0]*gd[1],gd[0]*gd[2],gd[1]*gd[2]};
                    doubleScalar wc=1.0/pointPtNum[pointI][pPointI];
                    vector ddx, ddy, ddz;
                    ddx = ddy = ddz = vector::zero;
                    for (label j = 0; j < 9; j++)
                    {
                        ddx += aSVD[0][j] * dd[j] * xx * wc;
                        ddy += aSVD[1][j] * dd[j] * xx * wc;
                        ddz += aSVD[2][j] * dd[j] * xx * wc;
                    }
                    UpGrad[pointI] += tensor(ddx, ddy, ddz);
                }
            }
            syncTools::syncPointList(mesh, UpGrad, plusEqOp<tensor>(), tensor::zero);
        }

        if(Q_Output)
        {
            forAll(Up, i)
            {
                Qp[i] = 0.5 * (sqr(tr(UpGrad[i])) - tr(((UpGrad[i]) & (UpGrad[i]))));
            }

            forAll(U, i)
            {
                Qc[i] = 0.5 * (sqr(tr(gradUc[i])) - tr(((gradUc[i]) & (gradUc[i]))));
            }
            Qc.correctBoundaryConditions();
            Qp.correctBoundaryConditions();
        }

        if(Omega_Output)
        {
            volTensorField S = 0.5 * (gradUc + T(gradUc));
            volTensorField W = 0.5 * (gradUc - T(gradUc));

            dimensionedScalar maxQc = max(Qc);
            doubleScalar e = maxQc.value() / 500;
            forAll(U, i)
            {
                OmegaC[i] = magSqr(W[i]) / (magSqr(W[i]) + magSqr(S[i]) + mag(e) + SMALL);
            }

            pointTensorField Sp = 0.5 * (UpGrad + T(UpGrad));
            pointTensorField Wp = 0.5 * (UpGrad - T(UpGrad));

            dimensionedScalar maxQp = max(Qp);
            e = maxQp.value() / 500;
            forAll(OmegaP, i)
            {
                OmegaP[i] = magSqr(Wp[i]) / (magSqr(Wp[i]) + magSqr(Sp[i]) + mag(e) + SMALL);
            }
            OmegaC.correctBoundaryConditions();
            OmegaP.correctBoundaryConditions();
        }

        // output

        // word header = "X,Y,Z,";
        // if(Q_Output)
        // {
        //     header += "Q,";
        // }
        // if(Omega_Output)
        // {
        //     header += "Omega,";
        // }
        // if(U_Output)
        // {
        //     header += "Ux,Uy,Uz,magU";
        // }

        std::ofstream outfile(runTime.timeName() + ".vtk");
        // outfile << header << endl;

        if (!outfile.is_open()) 
        {
            std::cerr << "Can not open the vtk file." << std::endl;
            return EXIT_FAILURE;
        }


        outfile << "# vtk DataFile Version 3.0\n";
        outfile << "CFD simulation data\n";
        outfile << "ASCII\n";
        outfile << "DATASET UNSTRUCTURED_GRID\n";

        if(Dimension==2)
        {
            outfile << "POINTS " << numVertex+noe << " double\n";
            for (label i = 0; i < mesh.nPoints() / 2; i++)
            {
                vector position = mesh.points()[i];
                outfile << position.x() << " " << position.y() << " " << 0.0 << std::endl;
            }
            for(label i = 0; i < mesh.nCells(); i++)
            {
                vector position = mesh.C()[i];
                outfile << position.x() << " " << position.y() << " " << 0.0 << std::endl;
            }

            label cellsEntrySize = 0;
            forAll(cellPointsType,i)
            {
                cellsEntrySize += cellPointsType[i].size();
            }

            outfile << "CELLS " << numElement << " " << cellsEntrySize << "\n";
            forAll(cellPointsType,i)
            {
                outfile << cellPointsType[i].size()-1;
                for(label j = 1; j < cellPointsType[i].size(); j++)
                {
                    outfile << " " << cellPointsType[i][j];
                }
                outfile << "\n";
            }

            outfile << "CELL_TYPES " << numElement << "\n";
            forAll(cellPointsType,i)
            {
                outfile << cellPointsType[i][0] << "\n";
            }

            if(Q_Output || Omega_Output || U_Output)
            {
                outfile << "POINT_DATA " << numVertex+noe << "\n";

                if (Q_Output)
                {
                    outfile << "SCALARS Q double 1\n";
                    outfile << "LOOKUP_TABLE default\n";
                    for (label i = 0; i < mesh.nPoints() / 2; i++)
                    {
                        outfile << Qp[i] << "\n";
                    }
                    for(label i = 0; i < mesh.nCells(); i++)
                    {
                        outfile << Qc[i] << "\n";
                    }
                }

                if (Omega_Output)
                {
                    outfile << "SCALARS Omega double 1\n";
                    outfile << "LOOKUP_TABLE default\n";
                    for (label i = 0; i < mesh.nPoints() / 2; i++)
                    {
                        outfile << OmegaP[i] << "\n";
                    }
                    for(label i = 0; i < mesh.nCells(); i++)
                    {
                        outfile << OmegaC[i] << "\n";
                    }
                }
                if(U_Output)
                {
                    outfile << "VECTORS Velocity double\n";
                    for (label i = 0; i < mesh.nPoints() / 2; i++)
                    {
                        outfile << Up[i][0] << " " << Up[i][1] << " " << Up[i][2] << "\n";
                    }
                    for(label i = 0; i < mesh.nCells(); i++)
                    {
                        outfile << Ud[i][0] << " " << Ud[i][1] << " " << Ud[i][2] << "\n";
                    }
                }
            }
            outfile.close();
            std::cout << "The VTK file has been created successfully: " + runTime.timeName() + ".vtk" << std::endl;
        }
        else if(Dimension==3)
        {
            outfile << "POINTS " << numVertex+noe << " double\n";
            for (label i = 0; i < mesh.nPoints(); i++)
            {
                vector position = mesh.points()[i];
                outfile << position.x() << " " << position.y() << " " << position.z() << std::endl;
            }
            for(label i = 0; i < mesh.nCells(); i++)
            {
                vector position = mesh.C()[i];
                outfile << position.x() << " " << position.y() << " " << position.z() << std::endl;
            }

            label cellsEntrySize = 0;
            forAll(cellPointsType,i)
            {
                cellsEntrySize += cellPointsType[i].size();
            }

            outfile << "CELLS " << numElement << " " << cellsEntrySize << "\n";
            forAll(cellPointsType,i)
            {
                outfile << cellPointsType[i].size()-1;
                for(label j = 1; j < cellPointsType[i].size(); j++)
                {
                    outfile << " " << cellPointsType[i][j];
                }
                outfile << "\n";
            }

            outfile << "CELL_TYPES " << numElement << "\n";
            forAll(cellPointsType,i)
            {
                outfile << cellPointsType[i][0] << "\n";
            }

            if(Q_Output || Omega_Output || U_Output)
            {
                outfile << "POINT_DATA " << numVertex+noe << "\n";

                if (Q_Output)
                {
                    outfile << "SCALARS Q double 1\n";
                    outfile << "LOOKUP_TABLE default\n";
                    for (label i = 0; i < mesh.nPoints(); i++)
                    {
                        outfile << Qp[i] << "\n";
                    }
                    for(label i = 0; i < mesh.nCells(); i++)
                    {
                        outfile << Qc[i] << "\n";
                    }
                }

                if (Omega_Output)
                {
                    outfile << "SCALARS Omega double 1\n";
                    outfile << "LOOKUP_TABLE default\n";
                    for (label i = 0; i < mesh.nPoints(); i++)
                    {
                        outfile << OmegaP[i] << "\n";
                    }
                    for(label i = 0; i < mesh.nCells(); i++)
                    {
                        outfile << OmegaC[i] << "\n";
                    }
                }

                if(U_Output)
                {
                    outfile << "VECTORS Velocity double\n";
                    for (label i = 0; i < mesh.nPoints(); i++)
                    {
                        outfile << Up[i][0] << " " << Up[i][1] << " " << Up[i][2] << "\n";
                    }
                    for(label i = 0; i < mesh.nCells(); i++)
                    {
                        outfile << Ud[i][0] << " " << Ud[i][1] << " " << Ud[i][2] << "\n";
                    }
                }
            }
            outfile.close();
            std::cout << "The VTK file has been created successfully: " + runTime.timeName() + ".vtk" << std::endl;
        }
        else
        {
            WarningInFunction << "The number of dimensions is wrong, it can only be 2D or 3D.";
            exit(1);
        }

    }

    Foam::Info << nl << "End" << nl << endl;
    return 0;
}