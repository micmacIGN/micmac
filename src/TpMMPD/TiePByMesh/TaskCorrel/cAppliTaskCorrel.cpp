#include "TaskCorrel.h"

cAppliTaskCorrel::cAppliTaskCorrel (
                                     cInterfChantierNameManipulateur * aICNM,
                                     const std::string & aDir,
                                     const std::string & aOri,
                                     const std::string & aPatImg
                                   ) :
    mICNM  (aICNM),
    mDir   (aDir),
    mOri   (aOri)
{
    vector<std::string> VNameImgs = *(aICNM->Get(aPatImg));
    for (uint aKI = 0; aKI<VNameImgs.size(); aKI++)
    {
        cImgForTiepTri* aImg = new cImgForTiepTri(*this, VNameImgs[aKI], aKI);
        mVImgs.push_back(aImg);
    }
    cout<<mVImgs.size()<<" images"<<endl;
}

void cAppliTaskCorrel::lireMesh(std::string & aNameMesh)
{
    InitOutil * aPly = new InitOutil (aNameMesh);
    mVTri = aPly->getmPtrListTri();
    cout<<mVTri.size()<<" triangles"<<endl;
}

cImgForTiepTri *cAppliTaskCorrel::DoOneTri(triangle * aTri)
{
    //reproject this tri in all img
    double cur_valElipse = DBL_MIN;
    cImgForTiepTri * imgMas = NULL;
    for (uint aKI = 0; aKI<mVImgs.size(); aKI++)
    {
        cTriForTiepTri * aTri2D = new cTriForTiepTri(*this, aTri);
        aTri2D->reprj(aKI);
        if (aTri2D->rprjOK())
        {
            //contraint ellipse
            double valElipse = aTri2D->valElipse();
            if (valElipse >= cur_valElipse)
            {
                cur_valElipse = valElipse;
                imgMas = mVImgs[aKI];
            }
        }
    }
    if (cur_valElipse != DBL_MIN && cur_valElipse > TT_SEUIL_RESOLUTION)
        return imgMas;
    else
        return NULL;
}

void cAppliTaskCorrel::DoAllTri()
{
    for (uint aKT=0; aKT<mVTri.size(); aKT++)
    {
        cImgForTiepTri * aImgMas = DoOneTri(mVTri[aKT]);
        cout<<aImgMas<<endl;
    }
}

