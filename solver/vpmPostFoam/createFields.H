    Info<< "Reading field U\n" << endl;
    volVectorField U
    (
        IOobject
        (
            "U",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );
    
    Info << "Reading field Up if present\n"<< endl;
    pointVectorField Up
    (
        IOobject
        (
            "Up",
            runTime.timeName(),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        pointMesh::New(mesh),
        dimensioned<vector>("zero",U.dimensions(),vector::zero)
    );
    #include "createPhi.H"

    volVectorField Ud(U);
    if(Dimension==2)
    {
        forAll(Ud,cellI)
        {
            labelList& pt=mixVertLabels[cellI];

            if(mixMeshType[cellI]==0)
            {
                // MCV2 for Qua element
                Ud[cellI]=(12.0*U[cellI]-(Up[pt[0]]+Up[pt[1]]+Up[pt[2]]+Up[pt[3]]))/8.0;
            }
            else
            {
                // MCV2 for Tri element
                Ud[cellI]=(12.0*U[cellI]-(Up[pt[0]]+Up[pt[1]]+Up[pt[2]]))/9.0;                 
            }
        }
        Ud.correctBoundaryConditions();
    }
    else if(Dimension==3)
    {
        forAll(Ud,cellI)
        {
            labelList& pt=mixVertLabels[cellI];

            switch (mixMeshType[cellI])
            {
                case 0:
                        // MCV2 for hex element
                        Ud[cellI]=(24.0*U[cellI]-(Up[pt[0]]+Up[pt[1]]+Up[pt[2]]+Up[pt[3]]+Up[pt[4]]+Up[pt[5]]+Up[pt[6]]+Up[pt[7]]))/16.0;
                        break;
                case 1:
                        //MCV2 for tet element
                        Ud[cellI]=(20.0*U[cellI]-(Up[pt[0]]+Up[pt[1]]+Up[pt[2]]+Up[pt[3]]))/16.0;
                        break;
                case 2:
                        //MCV2 for pyramid element
                        Ud[cellI]=(4948020.0*U[cellI]-233661.0*(Up[pt[0]]+Up[pt[1]]+Up[pt[2]]+Up[pt[3]])-311548.0*Up[pt[4]])/3701828.0;
                        break;
                case 3:
                        //MCV2 for prism element
                        Ud[cellI]=(18.0*U[cellI]-(Up[pt[0]]+Up[pt[1]]+Up[pt[2]]+Up[pt[3]]+Up[pt[4]]+Up[pt[5]]))/12.0;
                        break;
                default:
                        Info<<"polyMesh is not supported at current time!"<<endl;
                        exit(0);
            }
        }
        Ud.correctBoundaryConditions();
    }
    else
    {
        WarningInFunction << "The number of dimensions is wrong, it can only be 2D or 3D.";
        exit(1);
    }
    
    
