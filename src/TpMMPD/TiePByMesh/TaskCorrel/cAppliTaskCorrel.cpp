#include "TaskCorrel.h"


cParamAppliTaskCorrel::cParamAppliTaskCorrel(
                                             cInterfChantierNameManipulateur * aICNM,
                                             const std::string & aDir,
                                             const std::string & aOri,
                                             const std::string & aPatImg,
                                             bool & aNoTif,
                                             string  aMesureXML,
                                             bool aInverseOrder
                                            ):
    pICNM (aICNM),
    pDir (aDir),
    pOri (aOri),
    pPatImg (aPatImg),
    pNoTif (aNoTif),
    pMesureXML (aMesureXML),
    aInverseOrder (aInverseOrder)
{

}

//  ============================= **************** =============================
//  *                             cAppliTaskCorrel                             *
//  ============================= **************** =============================
cAppliTaskCorrel::cAppliTaskCorrel (cInterfChantierNameManipulateur * aICNM,
                                     const std::string & aDir,
                                     const std::string & aOri,
                                     const std::string & aPatImg,
                                     bool & aNoTif
                                   ) :
    mICNM  (aICNM),
    mDir   (aDir),
    mOri   (aOri),
    mNInter(0),
    mZoomF (1),
    mDirXML("XML_TiepTriNEW"),
    cptDel (0),
    mDistMax(TT_DISTMAX_NOLIMIT),
    mRech  (TT_DEF_SCALE_ZBUF),
    mNoTif (aNoTif),
    mKeepAll2nd (false),
    MD_SEUIL_SURF_TRIANGLE (TT_SEUIL_SURF_TRIANGLE),
    mWithGCP (false),
    mSafeZBuf (true),
    mZBuf_InverseOrder (false)
{
    ElTimer aChrono;
    cout<<"In constructor cAppliTaskCorrel : ";
    mVName = *(aICNM->Get(aPatImg));
    mVTask.resize(mVName.size());
    for (uint aKI = 0; aKI<mVName.size(); aKI++)
    {
        cImgForTiepTri* aImg = new cImgForTiepTri(this, mVName[aKI], int(aKI), mNoTif);
        ELISE_ASSERT(int(aKI) == aImg->Num(), "IN Constructor cAppli : Num() not coherence")
        mVImgs.push_back(aImg);
    }
    cout<<"Imgs creat "<<aChrono.uval()<<" sec" <<endl;
    aChrono.reinit();
    for (uint aKI = 0; aKI<mVImgs.size(); aKI++)
    {
        cImgForTiepTri* aImg = mVImgs[aKI];
        mVTask[aKI] = aImg->Task();
    }
    cout<<"Task creat "<<aChrono.uval()<<" sec" <<endl;
}


cAppliTaskCorrel::cAppliTaskCorrel (cInterfChantierNameManipulateur * aICNM,
                                     const std::string & aDir,
                                     const std::string & aOri,
                                     const std::string & aPatImg,
                                     bool & aNoTif,
                                     cParamAppliTaskCorrel *aParam
                                   ):
    cAppliTaskCorrel ( aICNM,
                       aDir,
                       aOri,
                       aPatImg,
                       aNoTif
                     )
{

    if (aParam->pMesureXML != "")
        ReadXMLMesurePts(aParam->pMesureXML, mVImgs);

    cout<<"Create Task with GCP..."<<endl;
    mVTaskWithGCP.resize(mVName.size());
    for (uint aKI = 0; aKI<mVImgs.size(); aKI++)
    {
        cImgForTiepTri* aImg = mVImgs[aKI];
        mVTaskWithGCP[aKI] = aImg->TaskWithGCP();
    }
    cout<<"Done !"<<endl;

    if (aParam->aInverseOrder)
    {
        this->mZBuf_InverseOrder = true;
    }
}

//  ============================= **************** =============================
//  *                             ReadXMLMesurePts                             *
//  ============================= **************** =============================
void cAppliTaskCorrel::ReadXMLMesurePts(string aGCPMesureXML, vector<cImgForTiepTri*> & mVImgs)
{
    cSetOfMesureAppuisFlottants aDico = StdGetFromPCP(aGCPMesureXML,SetOfMesureAppuisFlottants);

       std::list<cMesureAppuiFlottant1Im> & aLMAF = aDico.MesureAppuiFlottant1Im();

       std::cout<<"Reading mesure file..."<<std::flush;
       for (std::list<cMesureAppuiFlottant1Im>::iterator iT1 = aLMAF.begin() ; iT1 != aLMAF.end() ; iT1++)
       {

           std::list<cOneMesureAF1I> & aMes = iT1->OneMesureAF1I();
           string aNameIm = iT1->NameIm();
           cout<<endl<<" + Img : "<<aNameIm<<endl;
           cImgForTiepTri* aImg;
           for (uint akIm=0; akIm<mVImgs.size(); akIm++)
           {
               if (aNameIm == mVImgs[akIm]->Name())
               {
                   aImg = mVImgs[akIm];
                   break;
               }
           }
           aImg->Mesure().aNameIm = aNameIm;
           aImg->Mesure().aIndImg = aImg->Num();
           for (std::list<cOneMesureAF1I>::iterator iT2 = aMes.begin() ; iT2 != aMes.end() ; iT2++)
           {
               std::string aNamePt = iT2->NamePt();
               Pt2dr aPt = iT2->PtIm();
               cout<<"  + Pts : "<<aNamePt<<" "<<aPt<<endl;
               aImg->Mesure().aVPts.push_back(aPt);
               aImg->Mesure().aNamePt.push_back(aNamePt);
               aImg->ImgWithGCP() = true;
           }
       }
       std::cout<<"done!"<<std::endl;
       mWithGCP = true;
}

//  ============================= **************** =============================
//  *                             lireMesh                                     *
//  ============================= **************** =============================

void cAppliTaskCorrel::lireMesh(std::string & aNameMesh)
{
        cout<<"Lire mesh...";
        ElTimer aChrono;
        cMesh myMesh(aNameMesh, true);
        const int nFaces = myMesh.getFacesNumber();
        for (int aKTri=0; aKTri<nFaces; aKTri++)
        {
            cTriangle* aTri = myMesh.getTriangle(aKTri);
            vector<Pt3dr> aSm;
            aTri->getVertexes(aSm);
            cTri3D aTri3D (   aSm[0],
                              aSm[1],
                              aSm[2],
                              aKTri
                          );
            mVTriF.push_back(new cTriForTiepTri(this, aTri3D, aKTri));
            mVcTri3D.push_back(aTri3D);
        }
        mNameMesh = aNameMesh;
        cout<<"Finish - time "<<aChrono.uval()<<endl;
}

//  ============================= **************** =============================
//  *                             SetNInter                                    *
//  ============================= **************** =============================

void cAppliTaskCorrel::SetNInter(int & aNInter, double & aZoomF)
{
    mNInter = aNInter;
    mZoomF = aZoomF;
    if (mNInter > 1)
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

//  ============================= **************** =============================
//  *                             updateVTriFWithNewAppli                      *
//  ============================= **************** =============================

void cAppliTaskCorrel::updateVTriFWithNewAppli(vector<cTri3D> & tri)
{
    mVTriF.clear();
    for (int aKT=0; aKT<int(tri.size()); aKT++)
        mVTriF.push_back(new cTriForTiepTri(this, tri[aKT], aKT));
}

//  ============================= **************** =============================
//  *                             ZBuffer                                      *
//  ============================= **************** =============================
void cAppliTaskCorrel::ZBuffer()
{
    cout<<"Cal ZBuf && Tri Valid for each Img ...- NBImg : "<<mVName.size()<<endl;
    ElTimer aChrono;
    cParamZbufferRaster aParamZBuf;

    if (this->ZBuf_InverseOrder())
    {
        aParamZBuf.mInverseOrder = true;
    }

    cAppliZBufferRaster * aAppliZBuf = new cAppliZBufferRaster(
                                                                 mICNM,
                                                                 mDir,
                                                                 mOri,
                                                                 mVcTri3D,
                                                                 mVName,
                                                                 mNoTif,
                                                                 aParamZBuf
                                                              );


    aAppliZBuf->NInt() = 0;
    aAppliZBuf->DistMax() = mDistMax;
    aAppliZBuf->Reech() = mRech;
    aAppliZBuf->WithImgLabel() = true; //include calcul Image label triangle valab
    aAppliZBuf->SEUIL_SURF_TRIANGLE() = SEUIL_SURF_TRIANGLE();
    aAppliZBuf->Method() = MethodZBuf();
    aAppliZBuf->SetNameMesh(mNameMesh);
    aAppliZBuf->Param().mSafe = this->SafeZBuf();
    aAppliZBuf->DoAllIm(mVTriValid);


    ELISE_ASSERT(mVTriValid.size() == mVImgs.size(), "Sz VTriValid uncoherent Nb Img");

    // int cnt=0;  Warn unused
    for (uint aKIm=0; aKIm<mVImgs.size(); aKIm++)
    {
        cImgForTiepTri * aImg = mVImgs[aKIm];
        aImg->TriValid() = mVTriValid[aKIm];
    }
    delete aAppliZBuf;
    cout<<" - time : "<<aChrono.uval()<<endl;
}

//  ============================= **************** =============================
//  *                             DoOneTri                                     *
//  ============================= **************** =============================
cImgForTiepTri *cAppliTaskCorrel::DoOneTri(cTriForTiepTri *aTri2D)
{
    double cur_valElipse = -1.0;
    cImgForTiepTri * imgMas = NULL;
    Cur_Img2nd().clear();
    vector<Pt2dr> valEl_img;
    for (uint aKI = 0; aKI<mVImgs.size(); aKI++)
    {
        cImgForTiepTri * aImg = mVImgs[aKI];
        if (mNInter!=0) {cout<<" ++"<<aImg->Name();}
        if (aImg->TriValid()[aTri2D->Ind()])    //if image is invisible in this tri selon ZBuffer => skip
        {
            aTri2D->reprj(aImg);
            if (aTri2D->rprjOK())
            {
                //contraint ellipse
                double valElipse = aTri2D->valElipse(mNInter);
                if (mNInter!=0)
                {
                    cout<<" * "<<valElipse<<endl;
                }
                if (valElipse >= cur_valElipse)
                {
                    cur_valElipse = valElipse;
                    imgMas = aImg;
                }
                //Cur_Img2nd().push_back(int(aKI));
                valEl_img.push_back(Pt2dr(double(aKI),valElipse));
            }
            else
                {
                    if (mNInter!=0)
                    {
                        cout<<aTri2D->Pt1()<<aTri2D->Pt2()<<aTri2D->Pt3()<<" => reprojection error"<<endl;
                    }
            }
        }
        else
        {
            if (mNInter!=0)
            {
                cout<<aTri2D->Pt1()<<aTri2D->Pt2()<<aTri2D->Pt3()<<" => Non visible selon ZBuffer"<<endl;
            }
        }
    }
    sortDescendPt2drY(valEl_img);          // tri vecteur par valEllipse
    // critere pour choisir image master
    if (  valEl_img.size() > 1 &&          // il y a des images pour selectionner
          imgMas!=NULL &&                  // il y a au moins 1 val ellipse calcule
          cur_valElipse != -1.0 &&
          !std_isnan(cur_valElipse) &&     // => pourquoi il y a nan il val ellipse ? :(
          !std_isnan(valEl_img[0].y)
       )
    {
        //-----DEBUG----//
            std::ostringstream strs;
            strs <<  valEl_img[0].y;
            strs <<  " ";
            strs <<  cur_valElipse;
            std::string str = strs.str();
            string mesError = "val Ellipse not coherent valEl_img[0].y == cur_valElipse : " + str;
        ELISE_ASSERT( valEl_img[0].y == cur_valElipse, mesError.c_str());
        ELISE_ASSERT( int(valEl_img[0].x) == imgMas->Num(), "Num Img not coherent");
        //-----DEBUG----//
        if (mNInter!=0)
            cout<<endl;
        //-----Choisir les 2eme images par valEllipse----------
            if (valEl_img.size() > 4)
            {
                int get_lim = valEl_img.size();
                if (KeepAll2nd() == false)
                    get_lim = 4 + floor((double(valEl_img.size())-4.0)/2.0);
                for (int aK=1; aK<get_lim; aK++)
                    Cur_Img2nd().push_back(int(valEl_img[aK].x));
                imgMas = mVImgs[valEl_img[0].x];
                return imgMas;
            }
            else
            {
                for (uint aK=1; aK<(valEl_img.size()); aK++)
                    Cur_Img2nd().push_back(int(valEl_img[aK].x));
                imgMas = mVImgs[valEl_img[0].x];
                return imgMas;
            }
        //-----------------------------------------------------
    }
    else
    {
        if (mNInter!=0)
            cout<<"No master "<<"valElipse "<<cur_valElipse<<"  TT_SEUIL_RESOLUTION "<<TT_SEUIL_RESOLUTION<<endl<<endl;
        return NULL;
    }
    /*
    if (cur_valElipse != -1.0 && cur_valElipse > TT_SEUIL_RESOLUTION)
    {
        if (mNInter!=0)
            cout<<endl;
        return imgMas;
    }
    else
    {
        if (mNInter!=0)
            cout<<"No master "<<"valElipse "<<cur_valElipse<<"  TT_SEUIL_RESOLUTION "<<TT_SEUIL_RESOLUTION<<endl<<endl;
        return NULL;
    }
    */
}


//  ============================= **************** =============================
//  *                             GetIndTriHasGCP                              *
//  ============================= **************** =============================
void cAppliTaskCorrel::GetIndTriHasGCP()
{
    cout<<"Update index of Tri has GCP.."<<endl;
    for (uint aKImg=0; aKImg<mVImgs.size(); aKImg++)
    {
        cImgForTiepTri * aImg = mVImgs[aKImg];
        if (aImg->ImgWithGCP())
        {
            cout<<" + Img : "<<aImg->Name()<<endl;
            cout<<"  -> Found GCP : "<<endl;
            // read image TriLabel
            string fileTriLbl = mDir + "Tmp-ZBuffer/" + mNameMesh + "/" +aImg->Name() + "/" + aImg->Name() + "_TriLabel_DeZoom" +  ToString(int(1.0/mRech)) + ".tif";
            ELISE_ASSERT(ELISE_fp::exist_file(fileTriLbl),"Err GCP XML read in ZBuffer : File Img Label not found");

            Tiff_Im aImInd = Tiff_Im::StdConv(fileTriLbl);
            Im2D<double,double> mImgInd (aImInd.sz().x,aImInd.sz().y, -1.0);
            ELISE_COPY(mImgInd.all_pts(), aImInd.in(), mImgInd.out());

            for (uint aKGCP = 0; aKGCP<aImg->Mesure().aVPts.size(); aKGCP++)
            {
                Pt2dr aPtGCP = aImg->Mesure().aVPts[aKGCP];
                double aIndTri = mImgInd.GetR(Pt2di(aPtGCP*mRech));
                if (aIndTri == -1.0)
                {
                    vector<int> aVIndTri;
                    Pt2dr aPtGCP_Rech = aPtGCP*mRech;
                    // maybe this pixel of Zbuffer is in effet de bord
                    Pt2di aVois;
                    // take triangle autour (pixel autour rayon 4x4)
                    cout<<"  Srch Voisin : ";
                    for (aVois.x=-4; aVois.x<5; aVois.x++)
                    {
                        for (aVois.y=-4; aVois.y<5; aVois.y++)
                        {
                            double aIndTrii = aImInd.ReadIm().GetR(Pt2di(aPtGCP_Rech) + aVois);
                            cout<<" "<<(int)aIndTrii;
                            if (std::find(aVIndTri.begin(), aVIndTri.end(), (int)aIndTrii) != aVIndTri.end())
                            {
                                cout<<" "<<(int)aIndTrii;
                                aVIndTri.push_back((int)aIndTrii);
                            }
                        }
                    }
                    cout<<endl;
                    // check if GCP inside these triangle
                    if (aVIndTri.size() > 0)
                    {
                        for (uint i=0; i<aVIndTri.size(); i++)
                        {
                            cTriForTiepTri * aTri3D = mVTriF[aVIndTri[i]];
                            Pt2dr P1; Pt2dr P2; Pt2dr P3;
                            if(aTri3D->reprj_pure(aImg, P1, P2, P3))
                            {
                                // Check if  aPtGCP/mRech is inside P1, P2, P3

                                // if inside => break; update aIndTri

                            }
                        }
                    }
                }
                aImg->Mesure().aVIndTri.push_back((int)aIndTri);
                cout<<"   "<<aPtGCP<<" in Tri "<<aIndTri<<endl;
            }
        }
    }
    cout<<"done"<<endl;
}


//  ============================= **************** =============================
//  *                             DoAllTri                                     *
//  ============================= **************** =============================
void cAppliTaskCorrel::DoAllTri()
{
    cout<<"Nb Img: "<<mVImgs.size()<<" -Nb Tri: "<<mVTriF.size()<<endl;    
    for (uint aKT=0; aKT<mVcTri3D.size(); aKT++)
    {
        if (aKT % 100 == 0)
          cout<<"["<<(aKT*100.0/mVcTri3D.size())<<" %]"<<endl;
        cImgForTiepTri * aImgMas = DoOneTri(mVTriF[aKT]);
        cXml_Triangle3DForTieP aTaskTri;
        aTaskTri.P1() = mVcTri3D[aKT].P1();
        aTaskTri.P2() = mVcTri3D[aKT].P2();
        aTaskTri.P3() = mVcTri3D[aKT].P3();
        if (aImgMas != NULL && Cur_Img2nd().size() != 0)
        {
            if (mNInter!=0)
                cout<<"=> ImMas: "<<aImgMas->Tif().name()<<" - Num2nd : "<<Cur_Img2nd().size()<<endl<<endl;
            for (uint aKT2nd=0; aKT2nd<Cur_Img2nd().size(); aKT2nd++)
            {
                if (Cur_Img2nd()[aKT2nd] != aImgMas->Num() )
                {
                    if (Cur_Img2nd()[aKT2nd] < aImgMas->Num())
                        aTaskTri.NumImSec().push_back(Cur_Img2nd()[aKT2nd]);              
                    else
                        aTaskTri.NumImSec().push_back(Cur_Img2nd()[aKT2nd]-1);
                }
            }
            if (aTaskTri.NumImSec().size() != 0)
            {
                aImgMas->Task().Tri().push_back(aTaskTri);
                if (mWithGCP && aImgMas->ImgWithGCP())
                {
                    // if there is GCP inside this tri => store this tri as a task to correlation)
                    std::vector<int>::iterator it;
                    vector<int> aVTriHasGCPInImg = aImgMas->Mesure().aVIndTri;
                    it = find (aVTriHasGCPInImg.begin(), aVTriHasGCPInImg.end(), aKT);
                    size_t index = std::distance(aVTriHasGCPInImg.begin(), it);
                    if (it != aVTriHasGCPInImg.end())
                    {
                        std::cout << "DDDDDDDDD__ GCP : " << aImgMas->Mesure().aVPts[index] << " - tri : "<<aTaskTri.P1()<<aTaskTri.P2()<<aTaskTri.P3()<<'\n';
                        aImgMas->TaskWithGCP().Tri().push_back(aTaskTri);
                        mVTaskWithGCP[aImgMas->Num()].Tri().push_back(aTaskTri);
                    }
                }
                mVTask[aImgMas->Num()].Tri().push_back(aTaskTri);

//                if (xmlPairOut)
//                {
//                    //ajout XML couple output
//                    cCpleString aCpl( aImgMas->Name(), mVImgs[aTaskTri.NumImSec()[akIm2nd]]->Name() );
//                    mRelIm.Cple().push_back(aCpl);
//                }
            }
            else
                cptDel++;
        }
        else
            cptDel++;

    }
}

//  ============================= **************** =============================
//  *                             ExportXML                                    *
//  ============================= **************** =============================
void cAppliTaskCorrel::ExportXML(string aDirXML, Pt3dr clIni)
{
    cout<<"Write XML to "<<mICNM->Dir() + aDirXML.c_str() + "/"<<endl;
    ELISE_fp::MkDirSvp(aDirXML);
    if (mWithGCP)
        ELISE_fp::MkDirSvp(aDirXML+"_GCP");

    for (uint aKI=0; aKI<mVImgs.size(); aKI++)
    {
        cImgForTiepTri * aImg = mVImgs[aKI];
        //====this thing is eat RAM so much ======
        for (uint aKIi = 0; aKIi<mVImgs.size(); aKIi++)
        {
            if (aImg->Num() != int(aKIi))
            {
                aImg->Task().NameSec().push_back(mVImgs[aKIi]->Name());
                if (mWithGCP && aImg->ImgWithGCP())
                {
                    aImg->TaskWithGCP().NameSec().push_back(mVImgs[aKIi]->Name());
                    aImg->TaskWithGCP().NamePts() = aImg->Mesure().aNamePt;
                }
            }
        }
        //=========================================
        for (int aK=0 ; aK<2 ; aK++)
        {
            string fileXML = mICNM->Dir() + aDirXML + "/" + mVImgs[aKI]->Name() + std::string(aK==0 ? ".xml" : ".dmp");;
            MakeFileXML(aImg->Task(), fileXML);
        }
        if (mWithGCP && aImg->ImgWithGCP())
        {
            // write xml file when Image have GCP point
            string fileXML_WithGCP = mICNM->Dir() + aDirXML + "_GCP/" + mVImgs[aKI]->Name() + ".xml";
            MakeFileXML(aImg->TaskWithGCP(), fileXML_WithGCP);
        }
        //export mesh correspond with each image:
        DrawOnMesh aDraw;
        std::string fileMesh =  mICNM->Dir() + "PLYVerif/" + mVImgs[aKI]->Name() + ".ply";
        /*
        Pt3dr color(round(aKI*clIni.x/double(mVImgs.size())),
                    round(aKI*clIni.y/double(mVImgs.size())),
                    round(aKI*clIni.z/double(mVImgs.size())));
                    */
        Pt3dr color(
                    double( rand() % 255 ),
                    double( rand() % 255 ),
                    double( rand() % 255 )
                   );
        aDraw.drawListTriangle(aImg->Task().Tri(), fileMesh, color);
    }
//    if (xmlPairOut)
//    {
//        string fileXML = mICNM->Dir() + xmlNamePairOut + ".xml";
//        MakeFileXML(mRelIm,fileXML);
//    }
    cout<<"Del : "<<cptDel<<" /"<<this->VcTri3D().size()<<endl;
}

//  ============================= **************** =============================
//  *                           cAppliTaskCorrelByXML                          *
//  ============================= **************** =============================

cAppliTaskCorrelByXML::cAppliTaskCorrelByXML(const std::string & xmlFile,
                                                cInterfChantierNameManipulateur * aICNM,
                                                const std::string & aDir,
                                                const std::string & anOri,
                                                const std::string & aPatImg,
                                                const std::string & aPathMesh,
                                                bool aNoTif
                                             ):
          mICNM    (aICNM),
          mXmlFile (xmlFile),
          mDir     (aDir),
          mOri     (anOri),
          mPathMesh(aPathMesh),
          mNoTif   (aNoTif)

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
    cAppliTaskCorrel * aAppli = new cAppliTaskCorrel(aICNM, aDir, mOri, aFullPattern, mNoTif);

    //aAppli->updateVTriFWithNewAppli(mVcTri3D);
    aAppli->DoAllTri();
    cout<<aAppli->VTask().size()<<" task"<<endl;
    return aAppli->VTask();
}

void cAppliTaskCorrelByXML::DoAllCpl()
{
    cout<<"DoallCpl"<<endl;
    ELISE_ASSERT(mVCplImg.size()== mCplValidIndex.size(), "ERROR : Nb mVCplImg & Nb mCplValidIndex not coherent");
    for (uint aKCpl=0; aKCpl<mVCplImg.size(); aKCpl++)
    {
        CplString aCpl = mVCplImg[aKCpl];
        Pt2di aCplInd = mCplValidIndex[aKCpl];
        int indTask1 = aCplInd.x;
        int indTask2 = aCplInd.y;
        cout<<"Cpl "<<aKCpl<<" "<<aCpl.img1<<" "<<aCpl.img2<<" -Ind : "<<aCplInd<<endl;
        //lire mesh initialize:
        if (aKCpl == 0)
        {
            std::string aDir,aNameImg;
            std::string aFullPattern = aCpl.img1+"|"+aCpl.img2;
            SplitDirAndFile(aDir,aNameImg,aFullPattern);
            cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
            cAppliTaskCorrel * aAppli = new cAppliTaskCorrel(aICNM, aDir, mOri, aFullPattern, mNoTif);
            aAppli->lireMesh(mPathMesh/*, aAppli->VTri(), aAppli->VTriF()*/);
            mVTri = aAppli->VTri();
        }

        Cur_mVTask = DoACpl(aCpl);
        ELISE_ASSERT(Cur_mVTask.size()==2, "ERROR : Nb Task not coherent (1 cpl => 2 Task)");

        //Task1 => Mas=img1 , 2nd=img2
        cXml_TriAngulationImMaster aTaskImg1 =  Cur_mVTask[0];
        if (aTaskImg1.NameMaster() == aCpl.img1)
        {
            indTask1 = aCplInd.x;
            indTask2 = aCplInd.y;
        }
        else
        {
            indTask1 = aCplInd.y;
            indTask2 = aCplInd.x;
        }
        cout<<"Task: Mas= "<<aTaskImg1.NameMaster()<<" - NbTri: "<<aTaskImg1.Tri().size()<<" - Nb2nd: "<<aTaskImg1.NameSec().size()<<endl;
        ELISE_ASSERT(aTaskImg1.NameSec().size()==2, "ERROR : Nb img2nd not coherent (1 cpl => 2 img)");
        for (uint aKTgl = 0; aKTgl<aTaskImg1.Tri().size(); aKTgl++)
        {
            cXml_Triangle3DForTieP aTgl = aTaskImg1.Tri()[aKTgl];
            ELISE_ASSERT(aTgl.NumImSec().size()==1, "ERROR :NumImSec not coherent (must = 1)");
            aTgl.NumImSec()[0] = indTask2;
        }
        mVTask[indTask1].Tri().insert(mVTask[indTask1].Tri().end(), aTaskImg1.Tri().begin(), aTaskImg1.Tri().end());

        //Task2 => Mas=img2 , 2nd=img1
        cXml_TriAngulationImMaster aTaskImg2 =  Cur_mVTask[1];
        cout<<"Task: Mas= "<<aTaskImg2.NameMaster()<<" - NbTri: "<<aTaskImg2.Tri().size()<<" - Nb2nd: "<<aTaskImg2.NameSec().size()<<endl;
        ELISE_ASSERT(aTaskImg2.NameSec().size()==2, "ERROR : Nb img2nd not coherent (1 cpl => 2 img)");
        for (uint aKTgl = 0; aKTgl<aTaskImg2.Tri().size(); aKTgl++)
        {
            cXml_Triangle3DForTieP aTgl = aTaskImg2.Tri()[aKTgl];
            ELISE_ASSERT(aTgl.NumImSec().size()==1, "ERROR :NumImSec not coherent (must = 1)");
            aTgl.NumImSec()[0] = indTask1;
        }

        //fusion to collection task
        mVTask[indTask2].Tri().insert(mVTask[indTask2].Tri().end(), aTaskImg2.Tri().begin(), aTaskImg2.Tri().end());
        Cur_mVTask.clear();
    }
}

void cAppliTaskCorrelByXML::ExportXML(string & aXMLOut)
{
    cout<<"Write XML to "<<mICNM->Dir() + aXMLOut.c_str() + "/"<<endl;
    cout<<"Nb Task: "<<mVTask.size()<<endl;
    ELISE_fp::MkDirSvp(aXMLOut);
    for (uint aKTask=0; aKTask<mVTask.size(); aKTask++)
    {
        cXml_TriAngulationImMaster aTask = mVTask[aKTask];
        string fileXML = mICNM->Dir() + aXMLOut + "/" + aTask.NameMaster() + ".xml";
        MakeFileXML(aTask, fileXML);
    }
}





