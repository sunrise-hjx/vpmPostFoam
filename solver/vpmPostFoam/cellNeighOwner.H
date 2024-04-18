const labelUList& owner = mesh.owner();
const labelUList& neigh = mesh.neighbour();
vectorField cellNeigh(mesh.nFaces(),vector::zero);
vectorField cellOwner(mesh.nFaces(),vector::zero);
forAll(mesh.magSf(),faceI)
{
    label own = owner[faceI];
    label nei = neigh[faceI];
    cellNeigh[faceI]=mesh.C()[nei];
    cellOwner[faceI]=mesh.C()[own];
}

forAll(mesh.boundary(), patchI)
{
    const fvsPatchScalarField& magSfPf = mesh.magSf().boundaryField()[patchI];
    const pointField& faceCentres = mesh.faceCentres();
    const labelList& pfaceCells = mesh.boundary()[patchI].faceCells();
    forAll(magSfPf, pfaceI)
    {
        label pfCellI = pfaceCells[pfaceI];
        label faceI = mesh.boundary()[patchI].start()+pfaceI;
        if (magSfPf.coupled())
        {
            cellNeigh[faceI]=mesh.C()[pfCellI]-faceCentres[faceI];
            cellOwner[faceI]=mesh.C()[pfCellI]-faceCentres[faceI];                      
        }
        else
        {
            cellNeigh[faceI]=faceCentres[faceI];
            cellOwner[faceI]=mesh.C()[pfCellI]; 
        }
    }
}
syncTools::swapFaceList(mesh,cellNeigh);