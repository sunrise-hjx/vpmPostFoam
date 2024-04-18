
List<scalarField> pointPtNum(mesh.nPoints());
forAll(mesh.points(),pointI)
{
    const labelList& pPoints=mesh.pointPoints()[pointI];
    pointPtNum[pointI].setSize(pPoints.size());
    pointPtNum[pointI]=1.0;
}
scalarField edgePtNum(mesh.nEdges(),1.0);
syncTools::syncEdgeList(mesh, edgePtNum, plusEqOp<doubleScalar>(), 0.0);
forAll(edgePtNum,edgeI)
{
    if(edgePtNum[edgeI]>1)
    {
        label pointA=mesh.edges()[edgeI][0];
        label pointB=mesh.edges()[edgeI][1];

        const labelList& pAPoints=mesh.pointPoints()[pointA];
        forAll(pAPoints,pPointI)
        {
            if(pAPoints[pPointI]==pointB)
            {
                pointPtNum[pointA][pPointI]=edgePtNum[edgeI];
                break;
            }
        }

        const labelList& pBPoints=mesh.pointPoints()[pointB];
        forAll(pBPoints,pPointI)
        {
            if(pBPoints[pPointI]==pointA)
            {
                pointPtNum[pointB][pPointI]=edgePtNum[edgeI];
                break;
            }
        }
    }
}
edgePtNum.clear();