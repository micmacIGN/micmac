#include "TaskCorrel.h"

//  ============================= **************** =============================
//  *                             cAppliTaskCorrel                             *
//  ============================= **************** =============================
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
    mVTask.resize(VNameImgs.size());
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
        mVTask[aKI] = aImg->Task();
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

void cAppliTaskCorrel::lireMesh(std::string & aNameMesh, vector<triangle*> & tri, vector<cTriForTiepTri*> & triF)
{
    InitOutil * aPly = new InitOutil (aNameMesh);
    mVTri = aPly->getmPtrListTri();
    for (uint aKT=0; aKT<mVTri.size(); aKT++)
        mVTriF.push_back(new cTriForTiepTri(this, mVTri[aKT]));
}

void cAppliTaskCorrel::updateVTriFWithNewAppli(vector<triangle*> & tri)
{
    mVTri.clear();
    mVTri = tri;
    mVTriF.clear();
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
        //cout<<"No master "<<"valElipse "<<cur_valElipse<<"  TT_SEUIL_RESOLUTION "<<TT_SEUIL_RESOLUTION<<endl;
        return NULL;
    }
}

void cAppliTaskCorrel::DoAllTri()
{
    cout<<"Nb Img: "<<mVImgs.size()<<" -Nb Tri: "<<mVTriF.size()<<endl;
    for (uint aKT=0; aKT<mVTri.size(); aKT++)
    {
        //cout<<endl<<"Tri "<<aKT<<endl;
        cImgForTiepTri * aImgMas = DoOneTri(aKT);

        cXml_Triangle3DForTieP aTaskTri;
        aTaskTri.P1() = mVTri[aKT]->getSommet(0);
        aTaskTri.P2() = mVTri[aKT]->getSommet(1);
        aTaskTri.P3() = mVTri[aKT]->getSommet(2);
        if (aImgMas != NULL && Cur_Img2nd().size() != 0)
        {
            //cout<<" ++ImMas: "<<aImgMas->Tif().name()<<" - Num2nd : "<<Cur_Img2nd().size()<<endl;
            for (uint aKT2nd=0; aKT2nd<Cur_Img2nd().size(); aKT2nd++)
            {
                if (Cur_Img2nd()[aKT2nd] != aImgMas->Num() )
                    aTaskTri.NumImSec().push_back(Cur_Img2nd()[aKT2nd]);
            }
            if (aTaskTri.NumImSec().size() != 0)
            {
                aImgMas->Task().Tri().push_back(aTaskTri);
                mVTask[aImgMas->Num()].Tri().push_back(aTaskTri);
            }
            else
                cptDel++;
        }
        else
            cptDel++;
    }
}


void cAppliTaskCorrel::ExportXML(string aDirXML, Pt3dr clIni)
{
    cout<<"Write XML to "<<mICNM->Dir() + aDirXML.c_str() + "/"<<endl;
    ELISE_fp::MkDirSvp(aDirXML);
    for (uint aKI=0; aKI<mVImgs.size(); aKI++)
    {
        cImgForTiepTri * aImg = mVImgs[aKI];
        string fileXML = mICNM->Dir() + aDirXML + "/" + mVImgs[aKI]->Name() + ".xml";
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

//  ============================= **************** =============================
//  *                           cAppliTaskCorrelByXML                          *
//  ============================= **************** =============================

cAppliTaskCorrelByXML::cAppliTaskCorrelByXML(   const std::string & xmlFile,
                                                cInterfChantierNameManipulateur * aICNM,
                                                const std::string & aDir,
                                                const std::string & anOri,
                                                const std::string & aPatImg,
                                                const std::string & aPathMesh):
          mICNM    (aICNM),
          mXmlFile (xmlFile),
          mDir     (aDir),
          mOri     (anOri),
          mPathMesh(aPathMesh)

{
    mVNImgs = *(aICNM->Get(aPatImg));
    cout<<"Read Pattern : "<<mVNImgs.size()<<" images"<<endl;
    mVTask.resize(mVNImgs.size());
    for (uint aKI = 0; aKI<mVNImgs.size(); aKI++)
    {
        cXml_TriAngulationImMaster aTask;
        aTask.NameMaster() = mVNImgs[aKI];
        for (uint aKIi = 0; aKIi<mVNImgs.size(); aKIi++)
        {
            aTask.NameSec().push_back(mVNImgs[aKIi]);
        }
        mVTask[aKI] = aTask;
    }
    cAppliTaskCorrelByXML::importXML(mXmlFile);
    cAppliTaskCorrelByXML::filterCplProcess(mVCplImg, mVNImgs); //get cpl that have both imgs in pattern
    cout<<"Nb Cpl Valid: "<<mCplValidIndex.size()<<" cpls"<<endl;
}



void cAppliTaskCorrelByXML::importXML(string XmlFile)
{
    string line;
    ifstream file (XmlFile.c_str());
    if (file.is_open())
    {
        while ( getline (file,line) )
        {
            stringstream ss(line);
            string temp = "";
            getline(ss, temp, '>'); //get "<TagName"
            if (temp.find("Cple") != std::string::npos)
            {
                string cpl;
                getline(ss, cpl, '<');
                stringstream ssCpl(cpl);
                string nImg1, nImg2;
                getline(ssCpl, nImg1, ' ');
                getline(ssCpl, nImg2, '\n');
                CplString aCpl;
                aCpl.img1 = nImg1;
                aCpl.img2 = nImg2;
                mVCplImg.push_back(aCpl);
            }
        }
        file.close();
    }
    else cout << "Unable to open XML file";
    cout<<"ImportXML : "<<mVCplImg.size()<<" cpl"<<endl;
}

void cAppliTaskCorrelByXML::filterCplProcess(vector<CplString> & mVCplImg, vector<string> & mVNImgs)
{
    vector<CplString> cplValid;
    vector<Pt2di> cplIndex;
    for (uint aKCpl=0; aKCpl<mVCplImg.size(); aKCpl++)
    {
        CplString aCpl = mVCplImg[aKCpl];
        string nImg1 = aCpl.img1;
        string nImg2 = aCpl.img2;
        std::vector<string>::iterator itImg1;
        std::vector<string>::iterator itImg2;

        itImg1 = std::find(mVNImgs.begin(), mVNImgs.end(), nImg1);
        itImg2 = std::find(mVNImgs.begin(), mVNImgs.end(), nImg2);

        if  (   itImg1 != mVNImgs.end()
             && itImg2 != mVNImgs.end() )
        {
            mCplValidIndex.push_back(Pt2di(itImg1-mVNImgs.begin(), itImg2-mVNImgs.begin()));
            cplValid.push_back(aCpl);
        }
    }
    mVCplImg.clear();
    mVCplImg = cplValid;
}


vector<cXml_TriAngulationImMaster> cAppliTaskCorrelByXML::DoACpl(CplString aCpl)
{
    string nImg1 = aCpl.img1;
    string nImg2 = aCpl.img2;

    std::string aDir,aNameImg;
    std::string aFullPattern = nImg1+"|"+nImg2; //creat a fake pattern (contain just 2 img)
    SplitDirAndFile(aDir,aNameImg,aFullPattern);

    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    cAppliTaskCorrel * aAppli = new cAppliTaskCorrel(aICNM, aDir, mOri, aFullPattern);

    aAppli->updateVTriFWithNewAppli(mVTri);
    aAppli->DoAllTri();
    cout<<aAppli->VTask().size()<<" task"<<endl;
    return aAppli->VTask();
}

void cAppliTaskCorrelByXML::DoAllCpl()
{
    cout<<"DoallCpl"<<endl;
    for (uint aKCpl=0; aKCpl<mVCplImg.size(); aKCpl++)
    {
        CplString aCpl = mVCplImg[aKCpl];
        Pt2di aCplInd = mCplValidIndex[aKCpl];
        cout<<"Cpl "<<aKCpl<<" "<<aCpl.img1<<" "<<aCpl.img2<<" -Ind : "<<aCplInd<<endl;
        //lire mesh initialize:
        if (aKCpl == 0)
        {
            std::string aDir,aNameImg;
            std::string aFullPattern = aCpl.img1+"|"+aCpl.img2;
            SplitDirAndFile(aDir,aNameImg,aFullPattern);
            cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
            cAppliTaskCorrel * aAppli = new cAppliTaskCorrel(aICNM, aDir, mOri, aFullPattern);
            aAppli->lireMesh(mPathMesh, aAppli->VTri(), aAppli->VTriF());
            mVTri = aAppli->VTri();
        }
        Cur_mVTask = DoACpl(aCpl);
        for (uint aKTask=0; aKTask<Cur_mVTask.size(); aKTask++)
        {
            cXml_TriAngulationImMaster aTask =  Cur_mVTask[aKTask]; 
            cout<<"Task: Mas= "<<aTask.NameMaster()<<" - NbTri: "<<aTask.Tri().size()<<" - Nb2nd: "<<aTask.NameSec().size()<<endl;
            //change NumImSec in Tri
            for (uint aKTgl = 0; aKTgl<aTask.Tri().size(); aKTgl++)
            {
                cXml_Triangle3DForTieP aTgl = aTask.Tri()[aKTgl];
                aTgl.NumImSec()[0] = aCplInd.y;
            }
            //concate all Tri() in aTask to right position in mVTask
            mVTask[aCplInd.x].Tri().insert(mVTask[aCplInd.x].Tri().end(), aTask.Tri().begin(), aTask.Tri().end());
        }
        Cur_mVTask.clear();
    }
}

void cAppliTaskCorrelByXML::ExportXML(string & aXMLOut)
{
    cout<<"Write XML to "<<mICNM->Dir() + aXMLOut.c_str() + "/"<<endl;
    ELISE_fp::MkDirSvp(aXMLOut);
    for (uint aKTask=0; aKTask<mVTask.size(); aKTask++)
    {
        cXml_TriAngulationImMaster aTask = mVTask[aKTask];
        string fileXML = mICNM->Dir() + aXMLOut + "/" + aTask.NameMaster() + ".xml";
        MakeFileXML(aTask, fileXML);
    }
}





