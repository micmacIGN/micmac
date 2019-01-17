/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr


    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/
#include "StdAfx.h"
#include "cimgeo.h"
#include "cero_modelonepaire.h"
#include "cfeatheringandmosaicking.h"
#include "../mergehomol.h"
#include "ascii2tif.cpp"


extern int RegTIRVIS_main(int , char **);


class cLionPaw{
public:
    cLionPaw(int argc,char ** argv);
private:
    cInterfChantierNameManipulateur * mICNM;
    bool DoOri,DoMEC,Purge,mF;
    std::string mDir,mDirPat,mWD,mOut,mOutSufix;
    bool mRestoreTrash;
    int  mTPImSz1, mTPImSz2;
};

class cOneLionPaw{
public:
    cOneLionPaw(int argc,char ** argv);
    void testMTD();
    void SortImBlurred();
    void Restore();
private:
    cInterfChantierNameManipulateur * mICNM;
    bool DoOri,DoMEC,Purge,mF;
    std::string mDir,mDirPat,mWD,mOut,mOutSufix;
    std::list<std::string> mImName;
    std::map<int,std::string> mIms;
    bool mRestoreTrash;
    int  mTPImSz1, mTPImSz2;
};


cLionPaw::cLionPaw(int argc,char ** argv):
    DoOri(1),
    DoMEC(0),
    Purge(1),
    mF(0),
    mOutSufix("_MM"),
    mRestoreTrash(1),
    mTPImSz1(500),
    mTPImSz2(1000)
{
    mDir="./";
    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(mDir,"Working Directory", eSAM_IsDir)
                            << EAMC(mDirPat,"Directory Pattern to process", eSAM_IsPatFile)
                ,
                LArgMain()  << EAM(mOutSufix,"Suf",true, "resulting ply suffix , default result is Directory+'_MM'.ply .")
                            << EAM(DoMEC,"DoMEC",true, "Perform dense matching, def false .")
                            << EAM(DoOri,"DoOri",true, "Perform orientation, def true.")
                            << EAM(Purge,"Purge",true, "Purge intermediate results, def true.")
                            << EAM(mF,"F",true, "overwrite results, def false.")
                << EAM(mRestoreTrash,"Restore",true, "Restore images that are in the Poubelle folder prior to run the photogrammetric pipeline, def true.")
                << EAM(mTPImSz1,"TPImSz1",true, "Size of image to compute tie point at first iteration (prior to filter images), def=500")
                << EAM(mTPImSz2,"TPImSz2",true, "Size of image to compute tie point at second iteration (prior to compute orientation), def=1000")

                );
    if (!MMVisualMode)
    {
        mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);

        vector<std::string> aVDir = getDirListRegex(mDirPat);
        list<std::string> aLCom;

        for (auto & WD : aVDir){
        std::string aCom=MMBinFile(MM3DStr)+" TestLib AllAuto " + WD + " " + " Suf="+ mOutSufix + " DoMEC="+ToString(DoMEC)+ " DoOri=" + ToString(DoOri)
                + " Purge="+ToString(Purge)
                + " F="+ToString(mF)
                + " Restore="+ToString(mRestoreTrash)
                + " TPImSz1="+ToString(mTPImSz1)
                + " TPImSz2="+ToString(mTPImSz2)
                ;



        aLCom.push_back(aCom);
        std::cout << aCom << "\n";
        }
     cEl_GPAO::DoComInSerie(aLCom);
    }
}

cOneLionPaw::cOneLionPaw(int argc,char ** argv):
    DoOri(1),
    DoMEC(0),
    Purge(1),
    mF(0),
    mOutSufix("_MM"),
    mRestoreTrash(1),
    mTPImSz1(500),
    mTPImSz2(1000)

{
    mDir="./";
    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(mDir,"Working Directory", eSAM_IsDir)

                ,
                LArgMain() << EAM(mOutSufix,"Suf",true, "resulting ply suffix , default result is Directory+'_MM'.ply .")
                << EAM(DoMEC,"DoMEC",true, "Perform dense matching, def false .")
                << EAM(DoOri,"DoOri",true, "Perform orientation, def true.")
                << EAM(Purge,"Purge",true, "Purge intermediate results, def true.")
                << EAM(mF,"F",true, "overwrite results, def false.")
                << EAM(mRestoreTrash,"Restore",true, "Restore images that are in the Poubelle folder prior to run the photogrammetric pipeline, def true.")
                << EAM(mTPImSz1,"TPImSz1",true, "Size of image to compute tie point at first iteration (prior to filter images), def=500")
                << EAM(mTPImSz2,"TPImSz2",true, "Size of image to compute tie point at second iteration (prior to compute orientation), def=1000")

                );
    if (!MMVisualMode)
    {
     #if (ELISE_unix)

        // apericloud export
        mOut=mDir+mOutSufix+"_aero.ply";
        // pims2ply (Dense Cloud) export
        std::string mDC=mDir+mOutSufix+".ply";

        std::cout << "I will process data " << mDir << "\n";

        // martini ne fonctionne que si on est dans le directory grrr

        std::string aPat("'.*.(jpg|JPG)'");
        std::string aCom("");
        if (mRestoreTrash) Restore();
        // if no MTD, give fake ones
        testMTD();
        //if (DoOri) SortImBlurred();
        mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);

        if ( chdir(mDir.c_str())) {} // Warning retunr value

        if (DoOri){

            if(ELISE_fp::IsDirectory("Ori-C1") && mF==0){
               std::cout << "Orientation exist, use F=1 to overwrite it\n" ;
            }   else {

            ELISE_fp::PurgeDirRecursif("Ori-C1");

            aCom=MMBinFile(MM3DStr)+" Tapioca All "+ aPat + " " + ToString(mTPImSz1) + " Detect=Digeo";
            std::cout << aCom << "\n";
            system_call(aCom.c_str());
            aCom=MMBinFile(MM3DStr)+" Schnaps "+ aPat + " NbWin=200 MoveBadImgs=1 minPercentCoverage=60 VeryStrict=0 " ;
            std::cout << aCom << "\n";
            // iteratif
            for (int i(0) ; i<3 ; i++) {system_call(aCom.c_str());}

            aCom=MMBinFile(MM3DStr)+" Tapioca All "+ aPat + " " + ToString(mTPImSz2) + " Detect=Digeo";
            std::cout << aCom << "\n";
            system_call(aCom.c_str());

            aCom=MMBinFile(MM3DStr)+" Schnaps "+ aPat + " NbWin=200 MoveBadImgs=1 minPercentCoverage=60 VeryStrict=1 " ;
            std::cout << aCom << "\n";
            // iteratif
            for (int i(0) ; i<3 ; i++) {system_call(aCom.c_str());}

            aCom=MMBinFile(MM3DStr)+" Tapas RadialExtended "+ aPat + " Out=1 SH=_mini" ;
            std::cout << aCom << "\n";
            system_call(aCom.c_str());

            std::cout << aCom << "\n";
            //system_call(aCom.c_str());
            //aCom=MMBinFile(MM3DStr)+" AperiCloud "+ aPat + " C1 SH=-Ratafia Out=../cloud_" + mOut ;
            aCom=MMBinFile(MM3DStr)+" AperiCloud "+ aPat + " 1 Out=../" + mOut ;
            std::cout << aCom << "\n";
            system_call(aCom.c_str());
           }
        }

        if (DoMEC){
           if( ELISE_fp::IsDirectory("Ori-1")){

               aCom=MMBinFile(MM3DStr)+" PIMs BigMac " + aPat + " 1 ZoomF=8";
               std::cout << aCom << "\n";
               system_call(aCom.c_str());
               aCom=MMBinFile(MM3DStr)+" PIMs2Ply BigMac Out=" + mDC;
               std::cout << aCom << "\n";
               system_call(aCom.c_str());
           }
        }

        if (Purge) {

            std::list<std::string> aLDir;
            aLDir.push_back("Tmp-MM-Dir");
            aLDir.push_back("Pyram");
            aLDir.push_back("Pastis");
            aLDir.push_back("NewOriTmpQuick");
            aLDir.push_back("Tmp-ReducTieP");
            aLDir.push_back("Ori-RadialBasic");
            //aLDir.push_back("Ori-Martini");
            aLDir.push_back("Ori-InterneScan");
            aLDir.push_back("Homol_mini");

            for (auto & dir : aLDir){
            if(ELISE_fp::IsDirectory(dir))
            {
               std::cout << "Purge and remove directory " << dir << "\n";
               ELISE_fp::PurgeDirRecursif(dir);
               ELISE_fp::RmDir(dir);
            }
            }

        }
       #endif
    }
}

void cOneLionPaw::SortImBlurred(){

    vector<Pt2dr> aVPair;

    for (auto & imName : mIms){

    Im2D_REAL4 aIm=Im2D_REAL4::FromFileStd(imName.second);

    double aVar = VarLap(&aIm);
    Pt2dr aPair(imName.first, aVar);
    aVPair.push_back(aPair);
    }
    // choose the best image
    sortDescendPt2drY(aVPair);

    int imCt(0);
    for (auto & pair : aVPair){
    if (imCt<25) std::cout << imCt << ":  image " << mIms[round(pair.x)] << " i keep it \n";
    if (imCt>=25) {std::cout << imCt << ":  image " << mIms[round(pair.x)] << " i remove it \n";
    std::string aCom("mv " + mIms[round(pair.x)] + " " + mIms[round(pair.x)] + "_bu" );
    std::cout << aCom << "\n";
    system_call(aCom.c_str());
    }

    imCt++;
    }
}

void cOneLionPaw::Restore(){

    std::string aCom("mv "+ mDir + "/Poubelle/* "+ mDir + "/" );
    system_call(aCom.c_str());

}



                 void cOneLionPaw::testMTD(){

                 // il ne faut pas qu'il y en aie dans le répertroire "repetition1"
                 std::cout << "Test Metadata \n";
                 ELISE_fp::RmFileIfExist(mDir+"/MicMac-LocalChantierDescripteur.xml");

                 mICNM = cInterfChantierNameManipulateur::BasicAlloc("./");
                 std::string aPat(mDir+"/.*.(jpg|JPG)");
                 // get first image of im list

                 mImName = mICNM->StdGetListOfFile(aPat);

                 int imCt(0);
                 for (auto & Name : mImName){
                 mIms[imCt]=Name;
                 imCt++;

    }

                 cMetaDataPhoto aMTD =  cMetaDataPhoto::CreateExiv2(mIms[0]);
    if (aMTD.Foc35(true)<0){
        std::cout << "No metadata, I give some that are fake \n\n\n";
        std::string aCall("cp ../MicMac-LocalChantierDescripteur.xml "+mDir+"/");
        system_call(aCall.c_str());
    } else {
        std::cout << "Metadata found for image " << mIms[0]<< "\n\n";
    }

    }


// survey of a concrete wall, orientation very distorded, we export every tie point as GCP with a Z fixed by the user, in order to use them in campari
class cTPM2GCPwithConstantZ{
public:
    cTPM2GCPwithConstantZ(int argc,char ** argv);
private:
    cInterfChantierNameManipulateur * mICNM;
    bool mExpTxt,mDebug;
    double mZ;
    std::string mDir,mOriPat,mOut3D,mOut2D,mFileSH;
    std::list<std::string> mOriFL;// OriFileList
    cSetTiePMul * mTPM;
    std::vector<std::string> mImName;
    std::map<int, CamStenope*> mCams;
};

cTPM2GCPwithConstantZ::cTPM2GCPwithConstantZ(int argc,char ** argv)
{

    mOut2D="FakeGCP-2D.xml";
    mOut3D="FakeGCP-3D.xml";
    mDebug=0;
    mDir="./";
    mZ=0;
    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(mDir,"Working Directory", eSAM_IsDir)
                            << EAMC(mOriPat,"Orientation (xml) list of file", eSAM_IsPatFile)
                            << EAMC(mFileSH,"File of new set of homol format (PMulMachin).",eSAM_IsExistFile )
                ,
                LArgMain()
                << EAM(mZ,"Z",true, "Altitude to set for all tie points" )
                << EAM(mOut2D,"Out2D",true, "Name of resulting image measures file, def FakeGCP-2D.xml" )
                << EAM(mOut3D,"Out3D",true, "Name of resulting ground measures file, def FakeGCP-3D.xml" )
                << EAM(mDebug,"Debug",true, "Print message in terminal to help debugging." )
                );
    if (!MMVisualMode)
    {
        mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
        // load Tie point
        mTPM = new cSetTiePMul(0);
        mTPM->AddFile(mFileSH);
        // load orientations
        mOriFL = mICNM->StdGetListOfFile(mOriPat);
        std::string aKey= "NKS-Assoc-Ori2ImGen"  ;
        std::string aTmp1, aNameOri;
        for (auto &aOri : mOriFL){
            // retrieve name of image from name of orientation xml
            SplitDirAndFile(aTmp1,aNameOri,aOri);
            std::string NameIm = mICNM->Assoc1To1(aKey,aNameOri,true);
            mImName.push_back(NameIm);
            // retrieve IdIm
            cCelImTPM * ImTPM=mTPM->CelFromName(NameIm);
            if (ImTPM) {
            // map of Camera is indexed by the Id of Image (cSetTiePMul)
            mCams[ImTPM->Id()]=CamOrientGenFromFile(aOri,mICNM);
            } else {
            std::cout << "No tie points found for image " << NameIm << ".\n";
            }
        }

        // initialize dicco appui and mesureappui
        // 2D
        cSetOfMesureAppuisFlottants MAF;
        // 3D
        cDicoAppuisFlottant DAF;

        // loop on every config of TPM of the set of TPM
        int label(0);
        for (auto & aCnf : mTPM->VPMul())
        {
           // retrieve 3D position in model geometry
                std::vector<Pt3dr> aPts=aCnf->IntersectBundle(mCams);
                // add the points
                int aKp(0);
                for (auto & Pt: aPts){
                    // position 3D fake
                    Pt3dr PosXYZ(Pt.x,Pt.y,mZ);
                    cOneAppuisDAF GCP;
                    GCP.Pt()=PosXYZ;
                    GCP.NamePt()=std::string(to_string(label));
                    GCP.Incertitude()=Pt3dr(1.0,1.0,1.0);
                    DAF.OneAppuisDAF().push_back(GCP);

                    // position 2D
                    for (int nIm(0); nIm<aCnf->NbIm();nIm++)
                    {
                        int IdIm=aCnf->VIdIm().at(nIm);
                        cMesureAppuiFlottant1Im aMark;
                        aMark.NameIm()=mTPM->NameFromId(IdIm);
                        cOneMesureAF1I currentMAF;
                        currentMAF.NamePt()=std::string(to_string(label));
                        currentMAF.PtIm()= aCnf->GetPtByImgId(aKp,IdIm);
                        aMark.OneMesureAF1I().push_back(currentMAF);
                        MAF.MesureAppuiFlottant1Im().push_back(aMark);
                    }
                label++;  // total count of pt
                aKp++; // count of pt in config
                }
        }

         MakeFileXML(MAF,mOut2D);
         MakeFileXML(DAF,mOut3D);
        }
}

// we wish to improve coregistration between 2 orthos
class cCoreg2Ortho
{
    public:
    std::string mDir;
    cCoreg2Ortho(int argc,char ** argv);

    private:
    cImGeo * mO1;
    cImGeo * mO2;
    Im2D_REAL4 mO1clip,mO2clip;
    std::string mNameO1, mNameO2, mNameMapOut;
    Box2dr mBoxOverlapTerrain;

};
// vocation de test divers
cCoreg2Ortho::cCoreg2Ortho(int argc,char ** argv)
{

    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(mNameO1,"Name Ortho master", eSAM_IsExistFile)
                            << EAMC(mNameO2,"Name Ortho slave",eSAM_IsExistFile),
                LArgMain()  << EAM(mNameMapOut,"Out",true, "Name of resulting map")
                );

    if (!MMVisualMode)
    {

        mDir="./";
        mNameMapOut=mNameO2 +"2"+ mNameO1;
        //cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);

        if (ELISE_fp::exist_file(mNameO1) & ELISE_fp::exist_file(mNameO2))
        {

            // Initialise les 2 orthos
            mO1 = new cImGeo(mDir+mNameO1);
            mO2 = new cImGeo(mDir+mNameO2);

            // Dijkstra's single source shortest path algorithm

            mBoxOverlapTerrain=mO1->overlapBox(mO2);
            // clip les 2 orthos sur cette box terrain
            Im2D_REAL4 o1 = mO1->clipImTer(mBoxOverlapTerrain);
            Im2D_REAL4 o2 = mO2->clipImTer(mBoxOverlapTerrain);
            // determiner debut et fin de la ligne d'estompage

            Im2D_U_INT1 over(o1.sz().x,o2.sz().y,0);
            // carte des coût, varie de 0 à 1
            Im2D_REAL4 cost(o1.sz().x,o2.sz().y,1.0);
            // pixels d'overlap sont noté 1, pixel sans overlap sont noté 0
            ELISE_COPY(select(over.all_pts(),  o1.in()!=0 && o2.in()!=0),
                       1,
                       over.out());

            Tiff_Im::CreateFromIm(over,"Tmp_overlap.tif");

        } else { std::cout << "cannot find ortho 1 and 2, please check file names\n";}

    }
}


// appliquer une translation à une orientation

class cOriTran_Appli
{
    public:
    cOriTran_Appli(int argc,char ** argv);

    private:
    std::string mDir;
    std::string mFullDir;
    std::string mPat;
    std::string mOriIn, mOriOut;
    Pt3dr mTr;
};


cOriTran_Appli::cOriTran_Appli(int argc,char ** argv)
{
    ElInitArgMain
    (
    argc,argv,
        LArgMain()  << EAMC(mFullDir,"image pattern", eSAM_IsPatFile)
                    << EAMC(mOriIn,"Orientation Directory", eSAM_IsExistDirOri )
                    << EAMC(mTr,"Translation vector" )
                    << EAMC(mOriOut,"Orientation Out" )
        ,LArgMain()
    );


    if (!MMVisualMode)
    {

    SplitDirAndFile(mDir,mPat,mFullDir);
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
     StdCorrecNameOrient(mOriIn,mDir);
    const std::vector<std::string> aSetIm = *(aICNM->Get(mPat));

    // the bad way to do this, because I cannot find how to change the optical center of a camera with micmac classes
    std::string aNameTmp("Tmp-OriTrans.txt");
    std::string aCom = MMDir()
                + std::string("bin/mm3d")

                + " OriExport Ori-"
                +  mOriIn + std::string("/Orientation-")
                +  mPat + std::string(".xml")
                + std::string(" ") + aNameTmp;
    std::cout << aCom << "\n";

    system_call(aCom.c_str());

    aCom= MMDir()
            + std::string("bin/mm3d OriConvert '#F=N_X_Y_Z_W_P_K' ")
            +  aNameTmp + std::string(" ")
            + std::string("OriTrans")
            + std::string("  OffsetXYZ=") + ToString(mTr);

    std::cout << aCom << "\n";

    system_call(aCom.c_str());

    // je fais une bascule la dessus

    // je vérifie également avec un oriexport que l'offset fonctionnne

    aCom="cp Ori-"+mOriIn +"/AutoCal* Ori-"+mOriOut+"/";
    system_call(aCom.c_str());

   /* aCom= MMDir()
            + std::string("bin/mm3d GCP '#F=N_X_Y_Z_S_S_S' ")
            +  aNameTmp + std::string(" ")
            + std::string(mOriOut)
            + std::string("  OffsetXYZ=") + ToString(mTr);

    std::cout << aCom << "\n";

    system_call(aCom.c_str());

*/


    }

}
//    Applique une homographie à l'ensemble des images thermiques pour les mettres dans la géométrie des images visibles prises simultanément

class cTIR2VIS_Appli;
class cTIR2VIS_Appli
{
    public:
    void ReechThermicIm(std::vector<std::string> aPatImgs, std::string aHomog);
    void CopyOriVis(std::vector<std::string> aPatImgs, std::string aOri);
    cTIR2VIS_Appli(int argc,char ** argv);
    string T2V_imName(string tirName);
    string T2Reech_imName(string tirName);
    void changeImSize(std::vector<std::string> aLIm); //list image
    void changeImRadiom(std::vector<std::string> aLIm); //list image

    std::string mDir;
    private:
    std::string mFullDir;
    std::string mPat;
    std::string mHomog;
    std::string mOri;
    std::string mPrefixReech;
    bool mOverwrite;
    Pt2di mImSzOut;// si je veux découper mes images output, ex: homography between 2 sensors of different shape and size (TIR 2 VIS) but I want to have the same dimension as output
    Pt2di mRadiomRange;// If I want to change radiometry value, mainly to convert 16 bits to 8 bits
};


cTIR2VIS_Appli::cTIR2VIS_Appli(int argc,char ** argv) :
      mFullDir	("img.*.tif"),
      mHomog	("homography.xml"),
      mOri		("RTL"),
      mPrefixReech("Reech"),
      mOverwrite (false)



{
    ElInitArgMain
    (
    argc,argv,
        LArgMain()  << EAMC(mFullDir,"image pattern", eSAM_IsPatFile)
                    << EAMC(mHomog,"homography XML file", eSAM_IsExistFile ),
        LArgMain()  << EAM(mOri,"Ori",true, "ori name of VIS images", eSAM_IsExistDirOri )
                    << EAM(mOverwrite,"F",true, "Overwrite previous resampled images, def false")
                    << EAM(mImSzOut,"ImSzOut",true, "Size of output images")
                    << EAM(mRadiomRange,"RadiomRange",true, "range of radiometry of input images, if given, output will be 8 bits images")
    );


    if (!MMVisualMode)
    {

    SplitDirAndFile(mDir,mPat,mFullDir);
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
    const std::vector<std::string> aSetIm = *(aICNM->Get(mPat));


    ReechThermicIm(aSetIm,mHomog);

     if (EAMIsInit(&mOri))
     {
         StdCorrecNameOrient(mOri,mDir);
         mOri="Ori-"+mOri+"/";
         std::cout << "Copy orientation file." << std::endl;
         CopyOriVis(aSetIm,mOri);
      }

    // changer la taille des images out
    if (EAMIsInit(&mImSzOut))
    {
        //open first reech image just to read the dimension in order to print a message
        Tiff_Im mTif=Tiff_Im::StdConvGen(T2Reech_imName(aSetIm.at(0)),1,true);
        std::cout << "Change size of output images from " << mTif.sz() << " to " << mImSzOut << "\n";
        changeImSize(aSetIm);
    }

    // change the image radiometry
    if (EAMIsInit(&mRadiomRange))
    {
        std::cout << "Change images dynamic from range " << mRadiomRange << " to [0, 255] \n";
        changeImRadiom(aSetIm);
    }

    }
}



void cTIR2VIS_Appli::ReechThermicIm(
                                      std::vector<std::string> _SetIm,
                                      std::string aHomog
                                      )
{

     std::list<std::string>  aLCom;

    for(unsigned int aK=0; aK<_SetIm.size(); aK++)
    {
                string  aNameOut = "Reech_" + NameWithoutDir(StdPrefix(_SetIm.at(aK))) + ".tif";// le nom default donnée par ReechImMap

                std::string aCom = MMDir()
                            + std::string("bin/mm3d")
                            + std::string(" ")
                            + "ReechImMap"
                            + std::string(" ")
                            + _SetIm.at(aK)
                            + std::string(" ")
                            + aHomog;

                            if (EAMIsInit(&mPrefixReech)) {  aCom += " PrefixOut=" + T2Reech_imName(_SetIm.at(aK)) ; }

                            //+ " Win=[3,3]";// taille de fenetre pour le rééchantillonnage, par défaut 5x5

                bool Exist= ELISE_fp::exist_file(T2Reech_imName(_SetIm.at(aK)));

                if(!Exist || mOverwrite) {

                    std::cout << "aCom = " << aCom << std::endl;
                    //system_call(aCom.c_str());
                    aLCom.push_back(aCom);
                } else {
                    std::cout << "Reech image " << T2Reech_imName(_SetIm.at(aK)) << " exist, use F=1 to overwrite \n";
                }
    }
    cEl_GPAO::DoComInParal(aLCom);
}

// dupliquer l'orientation des images visibles de la variocam pour les images thermiques accociées
void cTIR2VIS_Appli::CopyOriVis(
                                      std::vector<std::string> _SetIm,
                                      std::string aOri
                                      )
{

    for(auto & imTIR: _SetIm)
    {
        //cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
        std::string aOriFileName(aOri+"Orientation-"+T2V_imName(imTIR)+".xml");
        if (ELISE_fp::exist_file(aOriFileName))
        {
        std::string aCom="cp " + aOriFileName + "   "+ aOri+"Orientation-" + T2Reech_imName(imTIR) +".xml";
        std::cout << "aCom = " << aCom << std::endl;
        system_call(aCom.c_str());
        } else
        {
        std::cout << "Can not copy orientation " << aOriFileName << " because file not found." << std::endl;
        }

    }
}



string cTIR2VIS_Appli::T2V_imName(string tirName)
{
   std::string visName=tirName;

   visName[0]='V';
   visName[2]='S';

   return visName;

}

string cTIR2VIS_Appli::T2Reech_imName(string tirName)
{
   return mPrefixReech+ "_" + tirName;
}



int T2V_main(int argc,char ** argv)
{
    cTIR2VIS_Appli aT2V(argc,argv);
    return EXIT_SUCCESS;
}


void cTIR2VIS_Appli::changeImSize(std::vector<std::string> aLIm)
{
    for(auto & imTIR: aLIm)
    {
    // load reech images
    Tiff_Im mTifIn=Tiff_Im::StdConvGen(T2Reech_imName(imTIR),1,true);
    // create RAM image
    Im2D_REAL4 im(mImSzOut.x,mImSzOut.y);
    // y sauver l'image
    ELISE_COPY(mTifIn.all_pts(),mTifIn.in(),im.out());
    // juste clipper
    Tiff_Im  aTifOut
             (
                 T2Reech_imName(imTIR).c_str(),
                 im.sz(),
                 GenIm::real4,
                 Tiff_Im::No_Compr,
                 Tiff_Im::BlackIsZero
             );
    // on écrase le fichier tif
   ELISE_COPY(im.all_pts(),im.in(),aTifOut.out());
    }
}

void cTIR2VIS_Appli::changeImRadiom(std::vector<std::string> aLIm)
{
    for(auto & imTIR: aLIm)
    {

    int minRad(mRadiomRange.x), rangeRad(mRadiomRange.y-mRadiomRange.x);

    // load reech images
    Tiff_Im mTifIn=Tiff_Im::StdConvGen(T2Reech_imName(imTIR),1,true);
    // create empty RAM image for imput image
    Im2D_REAL4 imIn(mTifIn.sz().x,mTifIn.sz().y);
    // create empty RAM image for output image
    Im2D_U_INT1 imOut(mTifIn.sz().x,mTifIn.sz().y);
    // fill it with tiff image value
    ELISE_COPY(
                mTifIn.all_pts(),
                mTifIn.in(),
                imIn.out()
               );

    // change radiometry
    for (int v(0); v<imIn.sz().y;v++)
    {
        for (int u(0); u<imIn.sz().x;u++)
        {
            Pt2di pt(u,v);
            double aVal = imIn.GetR(pt);
            unsigned int v(0);

            if(aVal!=0){
            if (aVal>minRad && aVal <minRad+rangeRad)
            {
                v=255.0*(aVal-minRad)/rangeRad;
            }
            }

            imOut.SetR(pt,v);
            //std::cout << "aVal a la position " << pt << " vaut " << aVal << ", transfo en " << v <<"\n";
        }
    }

    // remove file to be sure of result
    //ELISE_fp::RmFile(T2Reech_imName(imTIR));

    Tiff_Im aTifOut
             (
                 T2Reech_imName(imTIR).c_str(),
                 imOut.sz(),
                 GenIm::u_int1,
                 Tiff_Im::No_Compr,
                 Tiff_Im::BlackIsZero
             );
    // on écrase le fichier tif
   ELISE_COPY(imOut.all_pts(),imOut.in(),aTifOut.out());
    }
}









/*    comparaise des orthos thermiques pour déterminer un éventuel facteur de calibration spectrale entre 2 frame successif, expliquer pouquoi tant de variabilité spectrale est présente (mosaique moche) */
// à priori ce n'est pas ça du tout, déjà mauvaise registration TIR --> vis du coup les ortho TIR ne se superposent pas , du coup correction metrique ne peut pas fonctionner.
int CmpOrthosTir_main(int argc,char ** argv)
{
    std::string aDir, aPat="Ort_.*.tif", aPrefix="ratio";
    int aScale = 1;
    bool Test=true;
    std::list<std::string> mLFile;
    std::vector<cImGeo> mLIm;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aDir,"Ortho's Directory", eSAM_IsExistFile),
        LArgMain()  << EAM(aPat,"Pat",false,"Ortho's image pattern, def='Ort_.*'",eSAM_IsPatFile)
                    << EAM(aScale,"Scale",false,"Scale factor for both Orthoimages ; Def=1")
                    << EAM(Test,"T",false, "Test filtre des bords")
                    << EAM(aPrefix,"Prefix", false,"Prefix pour les ratio, default = ratio")

    );

    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    // create the list of images starting from the regular expression (Pattern)
    mLFile = aICNM->StdGetListOfFile(aPat);


    for (auto & imName : mLFile){
        //read Ortho

        cImGeo aIm(imName);
        std::cout << "nom image " << aIm.Name() << "\n";
        mLIm.push_back(aIm);
    }
    std::cout << mLFile.size() << " Ortho chargées.\n";

    //  tester si l'overlap est suffisant
    int i(0);
    for (auto aCurrentImGeo: mLIm)
    {
    i++;
    for (unsigned int j=i ; j<mLFile.size(); j++)
    {

        if (mLIm.at(j).Name()!=aCurrentImGeo.Name() && mLIm.at(j).overlap(&aCurrentImGeo,70))
        {

        Pt2di aTr=aCurrentImGeo.computeTrans(&mLIm.at(j));
        //std::string aName="ratio"+std::to_string(j)+"on"+std::to_string(j+1)+".tif";
        std::string aName=aPrefix+aCurrentImGeo.Name()+"on"+mLIm.at(j).Name()+".tif";
        // copie sur disque de l'image // pas très pertinent, je devrais plutot faire tout les calcul en ram puis sauver l'image à la fin avec un constructeur de cImGeo qui utilise une image RAM et les info du georef
        cImGeo        aImGeo(& aCurrentImGeo, aName);

        aImGeo.applyTrans(aTr);

        Im2D_REAL4 aIm=aImGeo.toRAM();
        Im2D_REAL4 aIm2(aIm.sz().x, aIm.sz().y);
        Im2D_REAL4 aImEmpty(aIm.sz().x, aIm.sz().y);
        Im2D_REAL4 aIm3=mLIm.at(j).toRAM();

        // l'image 1 n'as pas la meme taille, on la copie dans une image de meme dimension que l'im 0
        ELISE_COPY
                (
                    aIm3.all_pts(),
                    aIm3.in(),// l'image 1 n'as pas la meme taille, on la copie dans une image de meme dimension que l'im 0n(),
                    aIm2.oclip()
                    );

        // division de im 0 par im 1
        ELISE_COPY
                (
                    select(aIm.all_pts(),aIm2.in()>0),
                    (aIm.in())/(aIm2.in()),
                    aImEmpty.oclip()
                    );

        if (Test){
        // etape de dilation, effet de bord non désiré
        int it(0);
        do{

        Neighbourhood V8 = Neighbourhood::v8();
        Liste_Pts_INT2 l2(2);

        ELISE_COPY
        (
        dilate
        (
        select(aImEmpty.all_pts(),aImEmpty.in()==0),
        sel_func(V8,aImEmpty.in_proj()>0)
        ),
        1000,// je me fous de la valeur c'est pour créer un flux de points surtout
        aImEmpty.out() | l2 // il faut écrire et dans la liste de point, et dans l'image, sinon il va repecher plusieur fois le meme point
        );
        // j'enleve l'effet de bord , valleurs nulles
        ELISE_COPY
                (
                    l2.all_pts(),
                    0,
                    aImEmpty.out()
                    );

        it++;

        } while (it<3);

        }
        // je sauve mon image RAM dans mon image tif file
        aImGeo.updateTiffIm(&aImEmpty);

        // je calcule la moyenne du ratio
        int nbVal(0);
        double somme(0);
        for(int aI=0; aI<aImEmpty.sz().x; aI++)
        {
            for(int aJ=0; aJ<aImEmpty.sz().y; aJ++)
            {
                Pt2di aCoor(aI,aJ);
                double aValue = aImEmpty.GetR(aCoor);
                if (aValue!=0) {
                    somme +=aValue;
                    nbVal++;
                    //std::cout <<"Valeur:"<<aValue<< "\n";
                }
            }
            //fprintf(aFP,"\n");
        }
        somme/=nbVal;

        std::cout << "Ratio de l'image " << aCurrentImGeo.Name() << " sur l'image " << mLIm.at(j).Name() << "  caclulé, moyenne de  "<< somme << " ------------\n";
        // end if
        }
        // end boucle 1
    }
    // end boucle 2
    }
    return EXIT_SUCCESS;
}


int ComputeStat_main(int argc,char ** argv)
{
    double NoData(0);
    std::string aPat("Ree*.tif");
    std::list<std::string> mLFile;
    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(aPat,"Image pattern",eSAM_IsPatFile),
                LArgMain()  << EAM(NoData,"ND", "no data value, default 0")
                );

    std::cout <<" Debut\n";
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc("./");
    // create the list of images starting from the regular expression (Pattern)
    mLFile = aICNM->StdGetListOfFile(aPat);

    std::cout << mLFile.size() << " images\n";
    double aMax(0), aMin(0);

    for (auto & aName : mLFile){

    Tiff_Im mTif=Tiff_Im::StdConvGen(aName,1,true);
    Im2D_REAL4 aImRAM(mTif.sz().x, mTif.sz().y);
    ELISE_COPY
            (
                mTif.all_pts(),
                mTif.in(),
                aImRAM.out()
                );

    // je calcule la moyenne du ratio
    int nbVal(0);
    bool firstVal=1;
    double somme(0),min(1e30) /* MPD Warn uninit*/ ,max(0);
    for(int aI=0; aI<aImRAM.sz().x; aI++)
    {
        for(int aJ=0; aJ<aImRAM.sz().y; aJ++)
        {
            Pt2di aCoor(aI,aJ);
            double aValue = aImRAM.GetR(aCoor);
            if (aValue!=NoData) {
                if (firstVal)
                {
                    min=aValue;
                    firstVal=0;
                }
                if (aValue<min) min=aValue;
                if (aValue>max) max=aValue;
                somme +=aValue;
                nbVal++;
            }
        }
    }
    somme/=nbVal;
    std::cout <<"Statistique Image "<<aName<< "\n";
    std::cout << "Nb value !=" << NoData << " :" << nbVal << "\n";
    std::cout << "Mean :" << somme <<"\n";
    std::cout << "Max :" << max <<"\n";
    std::cout << "Min :" << min <<"\n";
    std::cout << "Dynamique (max-min) :" << max-min <<"\n";

    // stat sur toutes les images
    if (mLFile.front()==aName)
    {
        aMin=min;
        aMax=max;
    }

    if (max>aMax) aMax=max;
    if (min<aMin) aMin=min;

}
    std::cout << "Max de toutes les images :" << aMax <<"\n";
    std::cout << "Min de toutes les iamges :" << aMin <<"\n";
    std::cout << "Dynamique (max-min) :" << aMax-aMin <<"\n";

    return EXIT_SUCCESS;
}


// j'ai utilisé saisieAppui pour saisir des points homologues sur plusieurs couples d'images TIR VIS orienté
// je dois manipuler le résulat pour le tranformer en set de points homologues pour un unique couple d'images
// de plus, la saisie sur les im TIR est effectué sur des images rééchantillonnées, il faut appliquer une homographie inverse au points saisi
int TransfoMesureAppuisVario2TP_main(int argc,char ** argv)
{
    std::string a2DMesFileName, aOutputFile1, aOutputFile2,aImName("AK100419.tif"), aNameMap, aDirHomol("Homol-Man");

    ElInitArgMain
    (
    argc,argv,
    //mandatory arguments
    LArgMain()  << EAMC(a2DMesFileName, "Input mes2D file",  eSAM_IsExistFile)
                << EAMC(aNameMap, "Input homography to apply to TIR images measurements",  eSAM_IsExistFile),
    LArgMain()  << EAM(aImName,"ImName", true, "Name of Image for output files",  eSAM_IsOutputFile)
                << EAM(aOutputFile1,"Out1", true,  "Output TP file 1, def Homol-Man/PastisTIR_ImName/VIS_ImName.txt",  eSAM_IsOutputFile)
                << EAM(aOutputFile2,"Out2", true,  "Output TP file 2, def Homol-Man/PastisVIS_ImName/TIR_ImName.txt",  eSAM_IsOutputFile)
    );

    if (!EAMIsInit(&aOutputFile1)) {
        aOutputFile1=aDirHomol + "/PastisTIR_" + aImName + "/VIS_" + aImName + ".txt";
        if(!ELISE_fp::IsDirectory(aDirHomol)) ELISE_fp::MkDir(aDirHomol);
        if(!ELISE_fp::IsDirectory(aDirHomol + "/PastisTIR_" + aImName)) ELISE_fp::MkDir(aDirHomol + "/PastisTIR_" + aImName);
    }
    if (!EAMIsInit(&aOutputFile2)) {
        aOutputFile2=aDirHomol + "/PastisVIS_" + aImName + "/TIR_" + aImName + ".txt";
        if(!ELISE_fp::IsDirectory(aDirHomol)) ELISE_fp::MkDir(aDirHomol);
        if(!ELISE_fp::IsDirectory(aDirHomol + "/PastisVIS_" + aImName)) ELISE_fp::MkDir(aDirHomol + "/PastisVIS_" + aImName);
    }

    // lecture de la map 2D
    cElMap2D * aMap = cElMap2D::FromFile(aNameMap);

    // conversion de la map 2D en homographie; map 2D: plus de paramètres que l'homographie

    //1) grille de pt sur le capteur thermique auquel on applique la map2D
    ElPackHomologue  aPackHomMap2Homogr;
    for (int y=0 ; y<720; y +=10)
        {
         for (int x=0 ; x<1200; x +=10)
            {
             Pt2dr aPt(x,y);
             Pt2dr aPt2 = (*aMap)(aPt);
             ElCplePtsHomologues Homol(aPt,aPt2);
             aPackHomMap2Homogr.Cple_Add(Homol);
            }
        }
    // convert Map2D to homography
    cElHomographie H(aPackHomMap2Homogr,true);
    //H = cElHomographie::RobustInit(qual,aPackHomImTer,bool Ok(1),1, 1.0,4);

    // initialiser le pack de points homologues
    ElPackHomologue  aPackHom;

    cSetOfMesureAppuisFlottants aSetOfMesureAppuisFlottants=StdGetFromPCP(a2DMesFileName,SetOfMesureAppuisFlottants);

    int count=0;

    for( std::list< cMesureAppuiFlottant1Im >::const_iterator iTmes1Im=aSetOfMesureAppuisFlottants.MesureAppuiFlottant1Im().begin();
         iTmes1Im!=aSetOfMesureAppuisFlottants.MesureAppuiFlottant1Im().end();          iTmes1Im++    )
    {
        cMesureAppuiFlottant1Im anImTIR=*iTmes1Im;

        //std::cout<<anImTIR.NameIm().substr(0,5)<<" \n";
        // pour chacune des images thermique rééchantillonnée, recherche l'image visible associée
        if (anImTIR.NameIm().substr(0,5)=="Reech")
        {
            //std::cout<<anImTIR.NameIm()<<" \n";


            for (auto anImVIS : aSetOfMesureAppuisFlottants.MesureAppuiFlottant1Im()) {
            // ne fonctionne que pour la convention de préfixe Reech_TIR_ et VIS_
               if(anImTIR.NameIm().substr(10,anImTIR.NameIm().size()) == anImVIS.NameIm().substr(4,anImVIS.NameIm().size()))
               {
                   // j'ai un couple d'image.
                   //std::cout << "Couple d'images " << anImTIR.NameIm() << " et " <<anImVIS.NameIm() << "\n";

                   for (auto & appuiTIR : anImTIR.OneMesureAF1I())
                   {
                   //
                       for (auto & appuiVIS : anImVIS.OneMesureAF1I())
                       {
                       if (appuiTIR.NamePt()==appuiVIS.NamePt())
                       {
                           // j'ai 2 mesures pour ce point
                          // std::cout << "Pt " << appuiTIR.NamePt() << ", " <<appuiTIR.PtIm() << " --> " << appuiVIS.PtIm() << "\n";

                           // J'ajoute ce point au set de points homol
                           ElCplePtsHomologues Homol(appuiTIR.PtIm(),appuiVIS.PtIm());

                           aPackHom.Cple_Add(Homol);

                           count++;
                           break;
                       }
                       }
                   }
                   break;
               }
            }
       }

    // fin iter sur les mesures appuis flottant
    }
    std::cout << "Total : " << count << " tie points read \n" ;

    if (!EAMIsInit(&aOutputFile1) && !EAMIsInit(&aOutputFile2))
    {
    if(!ELISE_fp::IsDirectory(aDirHomol + "/PastisReech_TIR_" + aImName)) ELISE_fp::MkDir(aDirHomol + "/PastisReech_TIR_" + aImName);
    std::cout << "Homol pack saved in  : " << aDirHomol + "/PastisReech_TIR_" + aImName + "/VIS_" + aImName + ".txt" << " \n" ;
    aPackHom.StdPutInFile(aDirHomol + "/PastisReech_TIR_" + aImName + "/VIS_" + aImName + ".txt");
    aPackHom.SelfSwap();

    std::cout << "Homol pack saved in  : " << aDirHomol + "/PastisVIS_" + aImName + "/Reech_TIR_" + aImName + ".txt" << " \n" ;
    aPackHom.StdPutInFile(aDirHomol + "/PastisVIS_" + aImName + "/Reech_TIR_" + aImName + ".txt");

    }


    // appliquer l'homographie

    //aPackHom.ApplyHomographies(H.Inverse(),H.Id());
    aPackHom.ApplyHomographies(H,H.Id());
    // maintenant on sauve ce pack de points homologues
    std::cout << "Homol pack saved in  : " << aOutputFile1 << " \n" ;
    aPackHom.StdPutInFile(aOutputFile1);
    aPackHom.SelfSwap();
    std::cout << "Homol pack saved in  : " << aOutputFile2 << " \n" ;
    aPackHom.StdPutInFile(aOutputFile2);

	return EXIT_SUCCESS;
}


/* j'ai saisi des points d'appuis sur un vol 2 altitudes thermiques, j'aimerai voir si cette radiance est corrélée à
-Distance entre sensor et object  -->NON
-angle --> PAS TESTE
moins probable mais je teste quand même:
-position sur capteur --> NON
-temps écoulé depuis début du vol PAS TESTE
 */
int statRadianceVarioCam_main(int argc,char ** argv)
{
    std::string a2DMesFileName, a3DMesFileName, aOutputFile, aOri;

    ElInitArgMain
            (
                argc,argv,
                //mandatory arguments
                LArgMain()  << EAMC(a2DMesFileName, "Input mes2D file",  eSAM_IsExistFile)
                << EAMC(a3DMesFileName, "Input mes3D file",  eSAM_IsExistFile)
                << EAMC(aOri, "Orientation",  eSAM_IsExistDirOri),
                LArgMain()
                << EAM(aOutputFile,"Out", true,  "Output .txt file with radiance observation for statistic",  eSAM_IsOutputFile)

                );
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc("./");

    // open 2D measures
    cSetOfMesureAppuisFlottants aSetOfMesureAppuisFlottants=StdGetFromPCP(a2DMesFileName,SetOfMesureAppuisFlottants);
    // open 3D measures
    cDicoAppuisFlottant DAF= StdGetFromPCP(a3DMesFileName,DicoAppuisFlottant);
    std::list<cOneAppuisDAF> & aLGCP =  DAF.OneAppuisDAF();

    // create a map of GCP and position
    std::map<std::string, Pt3dr> aGCPmap;

    for (auto & GCP : aLGCP)
    {
        aGCPmap[GCP.NamePt()]=Pt3dr(GCP.Pt().x,GCP.Pt().y,GCP.Pt().z);
    }


    std::cout << "Image GCP U V rayon Radiance GroundDist \n" ;


    for( std::list< cMesureAppuiFlottant1Im >::const_iterator iTmes1Im=aSetOfMesureAppuisFlottants.MesureAppuiFlottant1Im().begin();
         iTmes1Im!=aSetOfMesureAppuisFlottants.MesureAppuiFlottant1Im().end();          iTmes1Im++    )
    {
        cMesureAppuiFlottant1Im anImTIR=*iTmes1Im;

        // open the image
        if (ELISE_fp::exist_file(anImTIR.NameIm()))
        {

            Tiff_Im mTifIn=Tiff_Im::StdConvGen(anImTIR.NameIm(),1,true);
            // create empty RAM image
            Im2D_REAL4 im(mTifIn.sz().x,mTifIn.sz().y);
            // fill it with tiff image value
            ELISE_COPY(
                        mTifIn.all_pts(),
                        mTifIn.in(),
                        im.out()
                        );
            // open the CamStenope
            std::string aNameOri=aICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+aOri+"@",anImTIR.NameIm(), true);
            CamStenope * aCam(CamOrientGenFromFile(aNameOri,aICNM));
            //std::cout << "Optical center Cam" << aCam->PseudoOpticalCenter() << "\n";
            // loop on the
            for (auto & appuiTIR : anImTIR.OneMesureAF1I())
            {
                // je garde uniquement les GCP dont le nom commence par L
                if (appuiTIR.NamePt().substr(0,1)=="L")
                   {
                    Pt3dr pt = aGCPmap[appuiTIR.NamePt()];
                   // std::cout << " Image " << anImTIR.NameIm() << " GCP " << appuiTIR.NamePt() << " ground position " << pt << " Image position " << appuiTIR.PtIm() << " \n";

                    double aRadiance(0);
                    int aNb(0);

                    Pt2di UV(appuiTIR.PtIm());
                    Pt2di sz(1,1);
                    ELISE_COPY(
                                rectangle(UV-sz,UV+Pt2di(2,2)*sz),// not sure how it work
                                Virgule(im.in(),1),
                                Virgule(sigma(aRadiance),sigma(aNb))
                                );
                    aRadiance/=aNb;
                    std::cout << " Radiance on windows of " << aNb << " px " << aRadiance << " \n";
                    aRadiance=aRadiance-27315;
                    // now determine incid and distance from camera to GCP

                    double aDist(0);
                    Pt3dr vDist=aCam->PseudoOpticalCenter()-pt;
                    aDist=euclid(vDist);

                    double aDistUV=euclid(appuiTIR.PtIm()-aCam->PP());


                    std::cout << anImTIR.NameIm() << " " << appuiTIR.NamePt() << " " << appuiTIR.PtIm().x   <<  " " << appuiTIR.PtIm().y  << " " << aDistUV << " " << aRadiance << " " << aDist << " \n";

                    }
                }

        }
        // fin iter sur les mesures appuis flottant
    }
	
	return EXIT_SUCCESS;
}


int MasqTIR_main(int argc,char ** argv)
{
    std::string aDir, aPat="Ort_.*.tif";
    std::list<std::string> mLFile;
    std::vector<cImGeo> mLIm;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aDir,"Ortho's Directory", eSAM_IsExistFile),
        LArgMain()  << EAM(aPat,"Pat",false,"Ortho's image pattern, def='Ort_.*'",eSAM_IsPatFile)

    );

    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    // create the list of images starting from the regular expression (Pattern)
    mLFile = aICNM->StdGetListOfFile(aPat);

    for (auto & aName : mLFile)
    {

    Tiff_Im im=Tiff_Im::StdConvGen("../OrthoTIR025/"+aName,1,true);

    std::string filenamePC= "PC"+aName.substr(3, aName.size()) ;

  /*
    //255: masqué. 0: ok
    Im2D_U_INT1 out(im.sz().x,im.sz().y);
    Im2D_REAL4 tmp(im.sz().x,im.sz().y);


    int minRad(27540), rangeRad(2546.0);


    ELISE_COPY
    (
    im.all_pts(),
    im.in(),
    tmp.out()
    );

    ELISE_COPY
    (
    select(tmp.all_pts(), tmp.in()>minRad && tmp.in()<minRad+rangeRad && tmp.in()!=0),
    255*(tmp.in()-minRad)/rangeRad,
    out.out()
    );

    ELISE_COPY
    (
    select(tmp.all_pts(), tmp.in()==0),
    0,
    out.out()
    );




    for (int v(0); v<tmp.sz().y;v++)
    {
        for (int u(0); u<tmp.sz().x;u++)
        {
            Pt2di pt(u,v);
            double aVal = tmp.GetR(pt);
            unsigned int v(0);

            if(aVal!=0){
            if (aVal>minRad && aVal <minRad+rangeRad)
            {
                v=255.0*(aVal-minRad)/rangeRad;
            }
            }

            out.SetR(pt,v);
            //std::cout << "aVal a la position " << pt << " vaut " << aVal << ", transfo en " << v <<"\n";
        }
    }


    std::cout << "je sauve l'image " << aName << "\n";
    ELISE_COPY
    (
        out.all_pts(),
        out.in(0),
        Tiff_Im(
            aName.c_str(),
            out.sz(),
            GenIm::u_int1,
            Tiff_Im::No_Compr,
            Tiff_Im::BlackIsZero
            ).out()
    );

*/
     Im2D_REAL4 masq(im.sz().x,im.sz().y);

    ELISE_COPY
    (
    im.all_pts(),
    im.in(0),
    masq.oclip()
    );


     std::cout << "détecte le bord pour image  " << filenamePC << "\n";
    int it(0);
    do{

    Neighbourhood V8 = Neighbourhood::v8();
    Liste_Pts_INT2 l2(2);

    ELISE_COPY
    (
    dilate
    (
    select(masq.all_pts(),masq.in()==0),
    sel_func(V8,masq.in_proj()>0)
    ),
    1000,// je me fous de la valeur c'est pour créer un flux de points surtout
    masq.oclip() | l2 // il faut écrire et dans la liste de point, et dans l'image, sinon il va repecher plusieur fois le meme point
    );
    // j'enleve l'effet de bord , valleurs nulles
    ELISE_COPY
            (
                l2.all_pts(),
                0,
                masq.oclip()
                );

    it++;

    } while (it<3);


/*
    // attention, écrase le ficher existant, pas propre ça
    std::cout << "je sauve l'image avec correction radiométrique " << aName << "\n";
    ELISE_COPY
    (
        masq.all_pts(),
        masq.in(0),
        Tiff_Im(
            aName.c_str(),
            masq.sz(),
            GenIm::int1,
            Tiff_Im::No_Compr,
            Tiff_Im::BlackIsZero
            ).out()
    );

*/
    ELISE_COPY
    (
    select(masq.all_pts(),masq.in()==0),
    255,
    masq.oclip()
    );

    ELISE_COPY
    (
    select(masq.all_pts(),masq.in()!=255),
    0,
    masq.oclip()
    );


    std::cout << "je sauve l'image les parties cachées " << filenamePC << "\n";
    ELISE_COPY
    (
        masq.all_pts(),
        masq.in(0),
        Tiff_Im(
            filenamePC.c_str(),
            masq.sz(),
            GenIm::u_int1,
            Tiff_Im::No_Compr,
            Tiff_Im::BlackIsZero
            ).out()
    );
    }
    return EXIT_SUCCESS;
}


int main_test2(int argc,char ** argv)
{
     //cORT_Appli anAppli(argc,argv);
     //CmpOrthosTir_main(argc,argv);
    //ComputeStat_main(argc,argv);
    //RegTIRVIS_main(argc,argv);  
    //MasqTIR_main(argc,argv);   
    //statRadianceVarioCam_main(argc,argv);
    //cTPM2GCPwithConstantZ(argc,argv);

   return EXIT_SUCCESS;
}

// launch all photogrammetric pipeline on a list of directory
int main_AllPipeline(int argc,char ** argv)
{
   cLionPaw(argc,argv);
   return EXIT_SUCCESS;
}
// launch a complete workflow on one image block
int main_OneLionPaw(int argc,char ** argv)
{
    cOneLionPaw(argc,argv);
    return EXIT_SUCCESS;
}

int main_testold(int argc,char ** argv)
{
 // manipulate the
    ofstream fout("/home/lisein/data/DIDRO/lp17/GPS_RGP/GNSS_pos/test.obs");
    ifstream fin("/home/lisein/data/DIDRO/lp17/GPS_RGP/GNSS_pos/tmp.txt");
    string line;

    std::string add("          40.000        40.000\n");
    add="\t40.000\t40.000\n";
      if (fin.is_open())
      {
        unsigned int i(0);
        while ( getline(fin,line) )
        {
          i++;

          if (i==17){
              i=0;
              fout << line << add;

          } else {

          if (i%2==1 && i>2)  {
               fout << line << add;
          } else {
              fout << line <<"\n";}
          }
        }

       } else { std::cout << "cannot open file in\n";}
        fin.close();
        fout.close();

   return EXIT_SUCCESS;
}





/*
 * // useless, Giang l'a codé en bien plus propre.
 * template <class T,class TB>
double VarLapl(Im2D<T,TB> * aIm,int aSzW);
//burriness is computed as variance of  Laplacian with a mean filter prior to reduce noise
//https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
template <class T,class TB>
double VarLapl(Im2D<T,TB> * aIm,int aSzW)
{
   Im2D_REAL4 aRes(aIm->sz().x,aIm->sz().y);

   Fonc_Num aF = aIm->in_proj();

   int aNbVois = ElSquare(1+2*aSzW);

   aF = rect_som(aF,aSzW) /aNbVois;

   double min,max;//  afin de pouvoir mettre le range de valeur entre 0 et 255
   ELISE_COPY(aIm->all_pts()
              ,aF
              ,aRes.out());

   double mean;
   int nb;

   ELISE_COPY(aRes.all_pts()
              ,Virgule(Laplacien(aRes.in_proj())                           ,1)
              ,Virgule(aRes.out()|sigma(mean)      ,sigma(nb)));

   std::cout << "mean of laplacian : " << mean << "\n";
   mean=mean/nb;

   std::cout << "nb of value: " << nb << "\n";
   // variance of laplacian
   Fonc_Num aFVar = ElSquare((aRes.in()-mean));

   double meanVarLap;
   ELISE_COPY(aRes.all_pts()
              ,aFVar
              ,sigma(meanVarLap));
   // mean of variance
   meanVarLap/=nb;
   return meanVarLap;
}
*/




/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'ucApplitilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe à
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/

