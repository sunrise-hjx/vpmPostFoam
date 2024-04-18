//Boundary Point_Search
    boolList isPatchPoint_(mesh.nPoints());
    isPatchPoint_ = false;

    forAll(mesh.boundaryMesh(), patchI)
    {
        const polyPatch& meshPatch = mesh.boundaryMesh()[patchI];

        if(!isA<emptyPolyPatch>(meshPatch) && !mesh.magSf().boundaryField()[patchI].coupled())
        {
            const labelList& pfaceCells = mesh.boundary()[patchI].faceCells();
            forAll(meshPatch, pfaceI)
            {
                label faceI = mesh.boundary()[patchI].start()+pfaceI;
                const face& facePoint=mesh.faces()[faceI];
                forAll(facePoint, facePointI) isPatchPoint_[facePoint[facePointI]] = true;
            }
        }
    }
    syncTools::syncPointList(mesh, isPatchPoint_, orEqOp<bool>(),false);