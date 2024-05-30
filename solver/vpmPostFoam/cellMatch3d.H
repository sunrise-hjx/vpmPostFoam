pyrMatcher pyr;
tetMatcher tet;
hexMatcher hex;
prismMatcher prism;
cellShape shape;
forAll(mixFaceLabels,cellI)
{
    bool isHex = hex.matches(mesh,cellI, shape);
    bool isTet = tet.matches(mesh,cellI, shape);
    bool isPyr = pyr.matches(mesh,cellI, shape);
    bool isPrism = prism.matches(mesh,cellI, shape);
    if(isHex)
    {
        mixMeshType[cellI]=0;
        mixFaceLabels[cellI]=hex.faceLabels();
        mixVertLabels[cellI]=hex.vertLabels();
    }
    else if(isTet)
    {
        mixMeshType[cellI]=1;
        mixFaceLabels[cellI]=tet.faceLabels();
        mixVertLabels[cellI]=tet.vertLabels();
    }
    else if(isPyr)
    {
        mixMeshType[cellI]=2;
        mixFaceLabels[cellI]=pyr.faceLabels();
        mixVertLabels[cellI]=pyr.vertLabels();
    }
    else if(isPrism)
    {
        mixMeshType[cellI]=3;
        mixFaceLabels[cellI]=prism.faceLabels();
        mixVertLabels[cellI]=prism.vertLabels();
    }
    else
    {
        Info<<"this scheme only support hex,tet,prism,pyramid mesh element for 3d!!"<<endl;
        exit(0);
    }
}