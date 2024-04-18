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

    
