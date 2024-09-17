
// labelListList facePoints(mesh.nFaces());
// forAll(mesh.faces(),faceI)
// {
//     const face& facePoint=mesh.faces()[faceI];
//     forAll(facePoint, facePointI)
//     {
//         label pI = facePoint[facePointI];
//         if (pI < mesh.nPoints() / 2)
//         {
//             facePoints[faceI].append(pI);
//         }
//     }
// }

// labelListList facePoints(mesh.nFaces());
// forAll(mesh.faces(),faceI)
// {
//     const face& facePoint=mesh.faces()[faceI];
//     forAll(facePoint, facePointI)
//     {
//         scalar pointZ=mesh.points()[facePoint[facePointI]].z();
//         scalar faceZ=mesh.faceCentres()[faceI].z();
//         if(pointZ<faceZ) facePoints[faceI].append(facePoint[facePointI]);
//     }
// }

labelListList facePoints(mesh.nFaces());
forAll(mesh.faces(),faceI)
{
    const face& facePoint=mesh.faces()[faceI];
    bool inPlane = true;
    forAll(facePoint, facePointI)
    {
        scalar pointZ=mesh.points()[facePoint[facePointI]].z();
        scalar faceZ=mesh.faceCentres()[faceI].z();
        if((pointZ-faceZ)>SMALL)
        {
            inPlane = false;
            break;
        }
    }
    if(inPlane)
    {
        forAll(facePoint, facePointI)
        {
            label pI = facePoint[facePointI];
            facePoints[faceI].append(pI);
        }
    }
    else
    {
        forAll(facePoint, facePointI)
        {
            label pI = facePoint[facePointI];
            if (pI < mesh.nPoints() / 2)
            {
                facePoints[faceI].append(pI);
            }
        }
    }
    
}