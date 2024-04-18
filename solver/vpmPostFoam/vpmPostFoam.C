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

int main(int argc, char *argv[])
{   
    timeSelector::addOptions(true, true);
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    #include "boundarySearch.H"

    #include "pointPtNum.H"

    #include "cellNeighOwner.H"

    #include "leastSqureSVD.H"

    #include "fields.H"

    #include "vpmPostDict.H"

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
        forAll(gradUc, cellI)
        {
            gradUc[cellI] = tensor(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            scalarRectangularMatrix &aSVDc = leastSquareDiffSVDc[cellI];
            const labelList& cPoints = mesh.cellPoints()[cellI];
            const labelList& cCells =  mesh.cellCells()[cellI];
            forAll(cCells, cCellI)
            {
                vector xx = U[cCells[cCellI]] - U[cellI];
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
                vector xx = Up[cPoints[cPointI]] - U[cellI];
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
        UpGrad = tensor::zero;
        forAll(Up, pointI)
        {
            if(isPatchPoint_[pointI]) continue;

            scalarRectangularMatrix& aSVD = leastSquareDiffSVD[pointI];
            const labelList& pCells=mesh.pointCells()[pointI];
            const labelList& pPoints=mesh.pointPoints()[pointI];
            forAll(pCells,pCellI)
            {
                vector xx = U[pCells[pCellI]] - Up[pointI];
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
        forAll(Up, i)
        {
            Qp[i] = 0.5 * (sqr(tr(UpGrad[i])) - tr(((UpGrad[i]) & (UpGrad[i]))));
        }

        forAll(U, i)
        {
            Qc[i] = 0.5 * (sqr(tr(gradUc[i])) - tr(((gradUc[i]) & (gradUc[i]))));
        }
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
        Qc.correctBoundaryConditions();
        Qp.correctBoundaryConditions();
        OmegaC.correctBoundaryConditions();
        OmegaP.correctBoundaryConditions();

        word header = "X,Y,Z,";
        if(Q_Output)
        {
            header += "Q,";
        }
        if(Omega_Output)
        {
            header += "Omega,";
        }
        if(U_Output)
        {
            header += "Ux,Uy,Uz,magU";
        }

        OFstream outfile(runTime.timeName() + ".txt");
        outfile << header << endl;
        forAll(Qp,i)
        {
            vector position = mesh.points()[i];
            outfile << position.x() << "," << position.y() << "," << position.z() << ",";
            if (Q_Output)
            {
                outfile << Qp[i] << ",";
            }
            if (Omega_Output)
            {
                outfile << OmegaP[i] << ",";
            }
            if (U_Output)
            {
                outfile << Up[i][0] << "," << Up[i][1] << "," << Up[i][2] << "," << mag(Up[i]);
            }
            outfile << endl;
        }
        forAll(Qc,i)
        {
            vector position = mesh.C()[i];
            outfile << position.x() << "," << position.y() << "," << position.z() << ",";
            if (Q_Output)
            {
                outfile << Qc[i] << ",";
            }
            if (Omega_Output)
            {
                outfile << OmegaC[i] << ",";
            }
            if (U_Output)
            {
                outfile << U[i][0] << "," << U[i][1] << "," << U[i][2] << "," << mag(U[i]);
            }
            outfile << endl;
        }
    }

    Foam::Info << nl << "End" << nl << endl;
    return 0;
}

