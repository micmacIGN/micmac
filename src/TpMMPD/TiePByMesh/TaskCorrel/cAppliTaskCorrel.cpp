#include "TaskCorrel.h"

//  ============================= cAppliTaskCorrel ==========================

cAppliTaskCorrel::cAppliTaskCorrel (
                                     cInterfChantierNameManipulateur * aICNM,
                                     const std::string & aDir,
                                     const std::string & aOri,
                                     const std::string & aPatImg
                                   ) :
    mICNM  (aICNM),
    mDir   (aDir),
    mOri   (aOri),
    mNInter(0),
    mZoomF (1),
    mDirXML("XML_TiepTriNEW"),
    cptDel (0)
{
    vector<std::string> VNameImgs = *(aICNM->Get(aPatImg));
    for (uint aKI = 0; aKI<VNameImgs.size(); aKI++)
    {
        cImgForTiepTri* aImg = new cImgForTiepTri(this, VNameImgs[aKI], aKI);
        mVImgs.push_back(aImg);
    }
    for (uint aKI = 0; aKI<mVImgs.size(); aKI++)
    {
        cImgForTiepTri* aImg = mVImgs[aKI];
        for (uint aKIi = 0; aKIi<mVImgs.size(); aKIi++)
        {
            aImg->Task().NameSec().push_back(mVImgs[aKIi]->Name());
        }
    }
}

void cAppliTaskCorrel::SetNInter(int & aNInter, double & aZoomF)
{
    mNInter = aNInter;
    mZoomF = aZoomF;
    if (mNInter != 0)
    {
        mVVW.resize(int(mVImgs.size()));
        for (uint aKI = 0; aKI<mVImgs.size(); aKI++)
        {
            cImgForTiepTri* aImg  = mVImgs[aKI];
            mVVW[aKI] = Video_Win::PtrWStd(Pt2di(mVImgs[aKI]->Sz()*mZoomF),true,Pt2dr(mZoomF,mZoomF));
            mVVW[aKI]->set_sop(Elise_Set_Of_Palette::TheFullPalette());
            mVVW[aKI]->set_title(mVImgs[aKI]->Tif().name());
            ELISE_COPY(mVVW[aKI]->all_pts(), aImg->Tif().in_proj(), mVVW[aKI]->ogray());
        }
        cout<<mVVW.size()<<" VW created"<<endl;
    }
}

void cAppliTaskCorrel::lireMesh(std::string & aNameMesh)
{
    InitOutil * aPly = new InitOutil (aNameMesh);
    mVTri = aPly->getmPtrListTri();
    for (uint aKT=0; aKT<mVTri.size(); aKT++)
        mVTriF.push_back(new cTriForTiepTri(this, mVTri[aKT]));
}

cImgForTiepTri *cAppliTaskCorrel::DoOneTri(int aNumT)
{
    double cur_valElipse = DBL_MIN;
    cImgForTiepTri * imgMas = NULL;
    cTriForTiepTri * aTri2D = mVTriF[aNumT];
    Cur_Img2nd().clear();
    for (uint aKI = 0; aKI<mVImgs.size(); aKI++)
    {
        cImgForTiepTri * aImg = mVImgs[aKI];
        aTri2D->reprj(aKI);
        if (aTri2D->rprjOK())
        {
            //contraint ellipse
            double valElipse = aTri2D->valElipse(mNInter);
            if (valElipse >= cur_valElipse)
            {
                cur_valElipse = valElipse;
                imgMas = aImg;
            }
            if (valElipse > TT_SEUIL_RESOLUTION)
                Cur_Img2nd().push_back(aKI);
        }
    }
    if (cur_valElipse != DBL_MIN && cur_valElipse > TT_SEUIL_RESOLUTION)
        return imgMas;
    else
    {
        cout<<"No master "<<"valElipse "<<cur_valElipse<<"  TT_SEUIL_RESOLUTION "<<TT_SEUIL_RESOLUTION<<endl;
        return NULL;
    }
}

void cAppliTaskCorrel::DoAllTri()
{
    for (uint aKT=0; aKT<mVTri.size(); aKT++)
    {
        cout<<endl<<"Tri "<<aKT<<endl;
        cImgForTiepTri * aImgMas = DoOneTri(aKT);

        cXml_Triangle3DForTieP aTaskTri;
        aTaskTri.P1() = mVTri[aKT]->getSommet(0);
        aTaskTri.P2() = mVTri[aKT]->getSommet(1);
        aTaskTri.P3() = mVTri[aKT]->getSommet(2);
        if (aImgMas != NULL && Cur_Img2nd().size() != 0)
        {
            cout<<" ++ImMas: "<<aImgMas->Tif().name()<<" - Num2nd : "<<Cur_Img2nd().size()<<endl;
            for (uint aKT2nd=0; aKT2nd<Cur_Img2nd().size(); aKT2nd++)
            {
                if (Cur_Img2nd()[aKT2nd] != aImgMas->Num() )
                    aTaskTri.NumImSec().push_back(Cur_Img2nd()[aKT2nd]);
            }
            if (aTaskTri.NumImSec().size() != 0)
                aImgMas->Task().Tri().push_back(aTaskTri);
            else
                cptDel++;
        }
        else
            cptDel++;
    }
}


void cAppliTaskCorrel::ExportXML(Pt3dr clIni)
{
    cout<<"Write XML to "<<mICNM->Dir() + mDirXML.c_str() + "/"<<endl;
    ELISE_fp::MkDirSvp(mDirXML);
    for (uint aKI=0; aKI<mVImgs.size(); aKI++)
    {
        cImgForTiepTri * aImg = mVImgs[aKI];
        string fileXML = mICNM->Dir() + mDirXML + "/" + mVImgs[aKI]->Name() + ".xml";
        MakeFileXML(aImg->Task(), fileXML);
        //export mesh correspond with each image:
        DrawOnMesh aDraw;
        std::string fileMesh =  mICNM->Dir() + "PLYVerif/" + mVImgs[aKI]->Name() + ".ply";
        Pt3dr color(round(aKI*clIni.x/double(mVImgs.size())),
                    round(aKI*clIni.y/double(mVImgs.size())),
                    round(aKI*clIni.z/double(mVImgs.size())));
        aDraw.drawListTriangle(aImg->Task().Tri(), fileMesh, color);
    }
    cout<<"Del : "<<cptDel<<endl;
}


