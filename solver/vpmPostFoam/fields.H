    volVectorField U0
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
    
    pointTensorField UpGrad
    (
        IOobject
        (
            "UpGrad",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        pointMesh::New(mesh),
        dimless
    );

    volScalarField Qc
    (
        IOobject
        (
            "Qc",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        0.5 * (sqr(tr(fvc::grad(U0))) - tr(((fvc::grad(U0)) & (fvc::grad(U0)))))
    );

    volScalarField OmegaC
    (
        IOobject
        (
            "OmegaC",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        Qc
    );

    pointScalarField Qp
    (
        IOobject
        (
            "Qp",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        pointMesh::New(mesh),
        dimless
    );

    pointScalarField OmegaP
    (
        IOobject
        (
            "OmegaP",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        pointMesh::New(mesh),
        dimless
    );

    forAll(Qp, i)
    {
        Qp[i] = 0.0;
        OmegaP[i] = 0.0;
    }
    forAll(Qc, i)
    {
        Qc[i] = 0.0;
        OmegaC[i] = 0.0;
    }