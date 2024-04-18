    List<scalarField> matrix45(mesh.nPoints());
    forAll(mesh.points(),pointI)
    {
        if(isPatchPoint_[pointI]) continue;
        const labelList& pCells=mesh.pointCells()[pointI];
        const labelList& pPoints=mesh.pointPoints()[pointI];

        matrix45[pointI].setSize(45);
        matrix45[pointI]=0.0;

        forAll(pCells,pCellI)
        {
            vector gd=mesh.C()[pCells[pCellI]]-mesh.points()[pointI];
            doubleScalar dd[9]={gd[0],gd[1],gd[2],gd[0]*gd[0],gd[1]*gd[1],gd[2]*gd[2],gd[0]*gd[1],gd[0]*gd[2],gd[1]*gd[2]};

            label iter=0;
            for(int n=0; n<9; n++)
            {
                for(int m=0; m<9; m++)
                {
                    if(n<=m)
                    {
                        matrix45[pointI][iter]+=dd[n]*dd[m];
                        iter++;
                    }
                }
            }
        }

        forAll(pPoints,pPointI)
        {
            doubleScalar wc=1.0/pointPtNum[pointI][pPointI];
            vector gd=mesh.points()[pPoints[pPointI]]-mesh.points()[pointI];
            doubleScalar dd[9]={gd[0],gd[1],gd[2],gd[0]*gd[0],gd[1]*gd[1],gd[2]*gd[2],gd[0]*gd[1],gd[0]*gd[2],gd[1]*gd[2]};

            label iter=0;
            for(int n=0; n<9; n++)
            {
                for(int m=0; m<9; m++)
                {
                    if(n<=m)
                    {
                        matrix45[pointI][iter]+=wc*dd[n]*dd[m];
                        iter++;
                    }
                }
            }
        }
    }
    syncTools::syncPointList(mesh, matrix45, plusEqOp<scalarField>(), matrix45[0]);

    List<scalarRectangularMatrix> leastSquareDiffSVD(mesh.nPoints());
    forAll(mesh.points(),pointI)
    {
        if(isPatchPoint_[pointI]) continue;
        scalarRectangularMatrix matrix(9,9);

        label iter=0;
        for(int n=0; n<9; n++)
        {
            for(int m=0; m<9; m++)
            {
                if(n<=m)
                {
                    matrix[n][m]=matrix45[pointI][iter];
                    iter++;
                }
                else matrix[n][m]=matrix[m][n];
            }
        }

        scalarRectangularMatrix invMatrix(9,9);
        invMatrix=SVDinv(matrix);
        leastSquareDiffSVD[pointI]=invMatrix;
    }

    List<scalarField> matrix45c(mesh.nCells());
    forAll(mesh.C(), cellI)
    {
        const labelList& cPoints = mesh.cellPoints()[cellI];
        const labelList& cCells =  mesh.cellCells()[cellI];
        matrix45c[cellI].setSize(45);
        matrix45c[cellI] = 0.0;
        forAll(cCells,cCellI)
        {
            vector gd=mesh.C()[cCells[cCellI]]-mesh.C()[cellI];
            doubleScalar dd[9]={gd[0],gd[1],gd[2],gd[0]*gd[0],gd[1]*gd[1],gd[2]*gd[2],gd[0]*gd[1],gd[0]*gd[2],gd[1]*gd[2]};

            label iter=0;
            for(int n=0; n<9; n++)
            {
                for(int m=0; m<9; m++)
                {
                    if(n<=m)
                    {
                        matrix45c[cellI][iter]+=dd[n]*dd[m];
                        iter++;
                    }
                }
            }
        }
        forAll(cPoints,cPointI)
        {
            vector gd=mesh.points()[cPoints[cPointI]]-mesh.C()[cellI];
            doubleScalar dd[9]={gd[0],gd[1],gd[2],gd[0]*gd[0],gd[1]*gd[1],gd[2]*gd[2],gd[0]*gd[1],gd[0]*gd[2],gd[1]*gd[2]};

            label iter=0;
            for(int n=0; n<9; n++)
            {
                for(int m=0; m<9; m++)
                {
                    if(n<=m)
                    {
                        matrix45c[cellI][iter]+=dd[n]*dd[m];
                        iter++;
                    }
                }
            }
        }
    }

    forAll(mesh.boundary(), patchI)
    {
        const polyPatch& meshPatch = mesh.boundaryMesh()[patchI];
        const labelList& pfaceCells = mesh.boundary()[patchI].faceCells();
        const pointField& faceCentres = mesh.faceCentres();
        if(mesh.magSf().boundaryField()[patchI].coupled())  
        {
            forAll(meshPatch, pfaceI)
            {
                label faceI = mesh.boundary()[patchI].start()+pfaceI;
                label pcellI = pfaceCells[pfaceI];
                vector gd = cellNeigh[faceI] - cellOwner[faceI];
                doubleScalar dd[9] = {gd[0],gd[1],gd[2],gd[0]*gd[0],gd[1]*gd[1],gd[2]*gd[2],gd[0]*gd[1],gd[0]*gd[2],gd[1]*gd[2]};
                label iter=0;
                for(int n=0; n<9; n++)
                {
                    for(int m=0; m<9; m++)
                    {
                        if(n<=m)
                        {
                            matrix45c[pcellI][iter]+=dd[n]*dd[m];
                            iter++;
                        }
                    }
                }
            }
        }
        else
        {
            forAll(meshPatch, pfaceI)
            {
                label faceI = mesh.boundary()[patchI].start()+pfaceI;
                label pcellI = pfaceCells[pfaceI];
                vector gd = faceCentres[faceI] - cellOwner[faceI];
                doubleScalar dd[9] = {gd[0],gd[1],gd[2],gd[0]*gd[0],gd[1]*gd[1],gd[2]*gd[2],gd[0]*gd[1],gd[0]*gd[2],gd[1]*gd[2]};
                label iter=0;
                for(int n=0; n<9; n++)
                {
                    for(int m=0; m<9; m++)
                    {
                        if(n<=m)
                        {
                            matrix45c[pcellI][iter]+=dd[n]*dd[m];
                            iter++;
                        }
                    }
                }
            }
        }
    }

    List<scalarRectangularMatrix> leastSquareDiffSVDc(mesh.nCells());
    forAll(mesh.C(), cellI)
    {
        scalarRectangularMatrix matrixc(9,9);

        label iter=0;
        for(int n=0; n<9; n++)
        {
            for (int m = 0; m < 9; m++)
            {
                if(n<=m)
                {
                    matrixc[n][m]=matrix45c[cellI][iter];
                    iter++;
                }
                else matrixc[n][m]=matrixc[m][n];
            }
        }
        scalarRectangularMatrix invMatrix(9,9);
        invMatrix=SVDinv(matrixc);
        scalarRectangularMatrix diffMatrix(3,9);
        for(int n=0; n<3; n++)
        {
            for(int m=0; m<9; m++)
            {
                diffMatrix[n][m] = invMatrix[n][m];
            }
        }
        leastSquareDiffSVDc[cellI] = diffMatrix;
    }