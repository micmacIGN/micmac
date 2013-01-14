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
// #include "XML_GEN/all_tpl.h"

/*
*/




class  cReadOri : public cReadObject
{
    public :
        cReadOri(char aComCar,const std::string & aFormat) :
               cReadObject(aComCar,aFormat,"S"),
               mPt(-1,-1,-1),
               mInc3(-1,-1,-1),
               mInc (-1)
        {
              AddString("N",&mName,true);
              AddPt3dr("XYZ",&mPt,true);
              AddPt3dr("WPK",&mWPK,false);
        }

        std::string mName;
        Pt3dr       mPt;
        Pt3dr       mWPK;
        Pt3dr       mInc3;
        double      mInc;
};



//================================================

cTxtCam::cTxtCam() :
   mCam (0),
   mOC  (0)
{
}

bool cCmpPtrCam::operator() (const cTxtCamPtr & aC1  ,const cTxtCamPtr & aC2)
{
    return aC1->mPrio < aC2->mPrio;
}




//================================================

                   // std::string  aKeyExport = "NKS-Assoc-Im2Orient@-" + VerifReexp;
                   // std::string aNOE = aICNM->Assoc1To1(aKeyExport,aNewCam.mNameIm,true);

class cAppli_Ori_Txt2Xml_main 
{
     public :
          cAppli_Ori_Txt2Xml_main (int argc,char ** argv);
     private :

         void ParseFile();
         void CalcImCenter();
         void DoTiePCenter();
         void CalVitesse(std::vector<cTxtCam *> & aVCam);

         std::string NameOrientation(const std::string &anOri,const cTxtCam & aCam)
         {
             return mICNM->Assoc1To1("NKS-Assoc-Im2Orient@-" +anOri,aCam.mNameIm,true);
         }

         std::string         mFilePtsIn;
         std::string         mOriOut;
         eTypeFichierOriTxt  mType;
         double              mDistNeigh;
         std::string         mKeyName2Image;
         bool                mAddCalib;
         std::string         mImC;
         int                 mNbImC;
         int                 mSizeC; 
         std::string         mReexpMatr;
         std::string         mDir;
         cInterfChantierNameManipulateur * mICNM;
         std::string         mFormat;
         char                mComment;
         cCalibrationInternConique mCIO;
         cChSysCo *          mCSC;
         cOrientationConique mOC0;
         std::string         mPatternIm;
         std::vector<cTxtCam *> mVCam;
         cTxtCam *            mCamC;
         std::string          mFileCalib;
         std::string          mPatImCenter;
         int                  mNbCam;
};


class cAttrArc
{
    public :
        double mCost;
};



typedef ElSom<cTxtCam *,cAttrArc>  tSom;


void cAppli_Ori_Txt2Xml_main::CalVitesse(std::vector<cTxtCam *> & aVCam)
{
   ElGraphe<cTxtCam *,cAttrArc> mGr;
   std::vector<tSom *> aVSom;

   int aNbC = aVCam.size();

   for (int aK= 0 ; aK<aNbC ; aK++)
   {
        tSom & aSom = mGr.new_som(aVCam[aK]);
        aVSom.push_back(&aSom);
   }

   for (int aK1=0 ; aK1<aNbC ; aK1++)
   {
       for (int aK2=aK1+1 ; aK2<aNbC ; aK2++)
       {
            tSom * aS1 = aVSom[aK1];
            tSom * aS2 = aVSom[aK2];
            cAttrArc anAttr;
            mGr.add_arc(*aS1,*aS2,anAttr);
       }
   }
}




cAppli_Ori_Txt2Xml_main::cAppli_Ori_Txt2Xml_main(int argc,char ** argv) :
    mType          (eOriTxtInFile),
    mDistNeigh     (0),
    mKeyName2Image ("NKS-Assoc-Id"),
    mAddCalib      (true),
    mImC           (""),
    mNbImC         (10),
    mSizeC         (1000),
    mReexpMatr     (),
    mDir           (),
    mICNM          (),
    mFormat        (""),
    mComment       (0),
    mCSC           (0),
    mPatternIm     (""),
    mCamC          (0),
    mNbCam         (0)
{

    bool Help;
    std::string aStrType; 
    if (argc >=2)
    {
        aStrType = argv[1];
        StdReadEnum(Help,mType,argv[1],eNbTypeOriTxt,true);

    }

    std::string aStrChSys;
    std::vector<std::string> aPrePost;

    ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAMC(aStrType,"Format specification") 
                      << EAMC(mFilePtsIn,"Orientation   File") 
                      << EAMC(mOriOut,"Targeted orientation") ,
           LArgMain() 
                      << EAM(aStrChSys,"ChSys",true,"Change coordinate file")
                      << EAM(mFileCalib,"Calib",true,"External XML calibration file")
                      << EAM(mAddCalib,"AddCalib",true,"Try to add calibration, def=true")
                      << EAM(aPrePost,"PrePost",true,"[Prefix,Postfix] to generate name of image from id")
                      << EAM(mKeyName2Image,"KN2I",true,"Key 2 compute Name Image from Id in file")
                      << EAM(mDistNeigh,"DN",true,"Neighbooring distance for Image Graphe")
                      << EAM(mImC,"ImC",true,"Image \"Center\" for computing AltiSol")
                      << EAM(mNbImC,"NbImC",true,"Number of neigboor around Image \"Center\" (Def=10)")
                      << EAM(mSizeC,"SizeSC",true,"Size of image to use for Tapioca for AltiSol (Def=1000)")
                      << EAM(mReexpMatr,"Reexp",true,"Reexport as Matrix (internal set up)")
    );

    if (EAMIsInit(&aPrePost))
    {
        ELISE_ASSERT(aPrePost.size()==2,"PrePost must be exactly of size 2");
        mKeyName2Image = "NKS-Assoc-AddPrePost@" + aPrePost[0] + "@"+aPrePost[1];
    }
    mDir = DirOfFile(mFilePtsIn);
    mICNM =  cInterfChantierNameManipulateur::BasicAlloc(mDir);



    if (mType==eOriTxtAgiSoft)
    {
         mFormat     = "N  X Y Z K W P";
         mComment    = '#';
    }
    else if (mType==eOriTxtInFile)
    {
       bool Ok = cReadObject::ReadFormat(mComment,mFormat,mFilePtsIn,true);
       ELISE_ASSERT(Ok,"File do not begin by format specification");
    }
    else
    {
        bool Ok = cReadObject::ReadFormat(mComment,mFormat,aStrType,false);
        ELISE_ASSERT(Ok,"Arg0 is not a valid format specif");
    }

    // cCalibrationInternConique aCIO;
    bool CalibIsInit = EAMIsInit(&mFileCalib) && mAddCalib;

     mCIO=  StdGetObjFromFile<cCalibrationInternConique>
            (
                  CalibIsInit ? (mDir + mFileCalib) : Basic_XML_MM_File("Template-Calib-Basic.xml"),
                  StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                  "CalibrationInternConique",
                  "CalibrationInternConique"
            );


    if (aStrChSys!="")
       mCSC = cChSysCo::Alloc(aStrChSys,"");


    mOC0 =   StdGetObjFromFile<cOrientationConique>
             (
                 Basic_XML_MM_File("Template-OrCamAng.xml"),
                 StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                 "OrientationConique",
                 "OrientationConique"
             );


    ParseFile();
    CalcImCenter();
    DoTiePCenter();


    std::cout << "PATC = " << mPatImCenter << "\n";
}



void cAppli_Ori_Txt2Xml_main::ParseFile()
{
    std::cout << "Comment=[" << mComment<<"]\n";
    std::cout << "Format=[" << mFormat<<"]\n";
    cReadOri aReadApp(mComment,mFormat);
    ELISE_fp aFIn(mFilePtsIn.c_str(),ELISE_fp::READ);

    char * aLine;

    while ((aLine = aFIn.std_fgets()))
    {
        if (aReadApp.Decode(aLine))
        {
           cTxtCam  & aNewCam = *(new cTxtCam);
           aNewCam.mNameIm = mICNM->Assoc1To1(mKeyName2Image,aReadApp.mName,true);

           if (1)  
           {
               std::cout << "Read data for " << aNewCam.mNameIm  <<  "  NB=" << mNbCam << "\n";
           }

           if (mNbCam==0)
              mPatternIm = aNewCam.mNameIm;
           else
              mPatternIm +=  "|"+aNewCam.mNameIm;

           std::string aNameOri = NameOrientation(mOriOut,aNewCam);

           if (mAddCalib && (! EAMIsInit(&mFileCalib) ))
           {
              const cMetaDataPhoto & aMTD = cMetaDataPhoto::CreateExiv2(mDir+aNewCam.mNameIm);
              Pt2di aSz =  aMTD.XifSzIm();
              Pt2dr aPP = Pt2dr(aSz) / 2.0;

              mCIO.PP() = aPP;
              mCIO.F() = aMTD.FocPix();
              mCIO.SzIm() = aSz;
              mCIO.CalibDistortion()[0].ModRad().Val().CDist() = aPP;
           }

           aNewCam.mOC = new cOrientationConique(mOC0);
           {
               Pt3dr aC =  aReadApp.mPt;
               if (mCSC)
                   aC = mCSC->Src2Cibl(aC);
               aNewCam.mOC->Externe().Centre() = aC;
               aNewCam.mC = aC;
           }

           // Calcul de la rotation
           if (aReadApp.IsDef(aReadApp.mWPK))
           {
              aNewCam.mOC->ConvOri().KnownConv().SetVal(eConvAngLPSDegre);
              aNewCam.mOC->Externe().ParamRotation().CodageAngulaire().SetVal(aReadApp.mWPK);
           }

           if (mAddCalib)
           {
              aNewCam.mOC->Interne().SetVal(mCIO);
           }
           MakeFileXML(*(aNewCam.mOC),aNameOri);


           aNewCam.mCam = CamOrientGenFromFile(aNameOri,mICNM);

           if (EAMIsInit(&mReexpMatr))
           {
              cOrientationConique aOriEx = aNewCam.mCam->StdExportCalibGlob();
              MakeFileXML(aOriEx,NameOrientation(mReexpMatr,aNewCam));
           }

           if (1)
           {
               if (!mVCam.empty())
               {
                   cTxtCam * aPrec = mVCam.back();
                   Pt3dr aV3 = aNewCam.mC - aPrec->mC;
                   Pt2dr aV2 (aV3.x,aV3.y);

                   double aTeta = angle(aV2);
                   aTeta = aTeta * (180/PI);
                   if (aTeta <0) aTeta += 360;
                   std::cout  << "Annagle " << aTeta << " " << aReadApp.mWPK.z << "\n";
               }
           }

           mVCam.push_back(&aNewCam);
           if (EAMIsInit(&mImC) &&  (aNewCam.mNameIm == mImC))
           {
              mCamC = &aNewCam;
           }

           mNbCam++;
        }
    }
    aFIn.close();
}

void cAppli_Ori_Txt2Xml_main::CalcImCenter()
{
    if (mCamC==0) return;

    for (int aK1=0 ; aK1<mNbCam ; aK1++)
    {
        cTxtCam & aC1 = *(mVCam[aK1]);
        aC1.mPrio = euclid(aC1.mC-mCamC->mC);
    }
    cCmpPtrCam aCmp;
    std::sort(mVCam.begin(),mVCam.end(),aCmp);
    for (int aK=0 ; aK<mNbImC ; aK++)
    {
        mVCam[aK]->mSelC = true;
        if (aK!=0)
        {
           mPatImCenter += "|";
        }
        mPatImCenter += mVCam[aK]->mNameIm;
    }
}

void cAppli_Ori_Txt2Xml_main::DoTiePCenter()
{
    if (mCamC==0) return;

    std::string aCom =    MM3DStr
                            + std::string(" Tapioca All \"") 
                            + mPatImCenter
                            + std::string("\" ")
                            + ToString(mSizeC) ;

    system_call(aCom.c_str());
    // std::cout << "PATC = " << aPatC << "\n";
}

/*
    cSauvegardeNamedRel  aRelIm;    
    if (EAMIsInit(&aDistNeigh))
    {
        for (int aK1=0 ; aK1<aNbCam ; aK1++)
        {
            for (int aK2=0 ; aK2<aNbCam ; aK2++)
            {
                if (aK1 != aK2)
                {
                    const cTxtCam & aC1 = *(aVCam[aK1]);
                    const cTxtCam & aC2 = *(aVCam[aK2]);
                    Pt3dr aP1 = aC1.mCam->PseudoOpticalCenter();
                    Pt3dr aP2 = aC2.mCam->PseudoOpticalCenter();

                    if (euclid(aP1-aP2)<aDistNeigh)
                    {
                        cCpleString aCpl(aC1.mNameIm,aC2.mNameIm);
                        aRelIm.Cple().push_back(aCpl);
                    }
                }
            }
        }
         MakeFileXML(aRelIm,aDir+aNameCpl);
    }

    // MakeFileXML(aDico,aFilePtsOut);
*/

//================================================

int Ori_Txt2Xml_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv,3);
    cAppli_Ori_Txt2Xml_main(argc,argv);
    return 0;
}





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

A cet égard  l'attention de l'utilisateur est attirée sur les risques
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
