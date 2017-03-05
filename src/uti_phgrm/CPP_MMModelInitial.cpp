/*Header-MicMac-eLiSe-25/06/2007peroChImMM_main

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

/*    Un filtrage basique qui supprime recursivement (par CC) les points obtenus a une resol
 non max et voisin du vide
*/

#define TheNbPref 5


/**************************************************************************/
/*                                                                        */
/*                                                                        */
/*                                                                        */
/**************************************************************************/

// cMMByImNM  :  MicMac By Image Name Manipulator


void MakeListofName(const std::string & aFile,const cInterfChantierNameManipulateur::tSet  *aSet)
{
   cListOfName aLON;
   std::copy(aSet->begin(),aSet->end(),back_inserter(aLON.Name()));

   MakeFileXML(aLON,aFile);
}


void AddistofName(const std::string & aFile,const cInterfChantierNameManipulateur::tSet  *aS0)
{
    std::set<std::string> aSetFin;

    for (int aK=0 ; aK<int(aS0->size()) ; aK++)
        aSetFin.insert((*aS0)[aK]);

    if (ELISE_fp::exist_file(aFile))
    {
        cListOfName aL0 =  StdGetFromPCP(aFile,ListOfName);
        for
        (
           std::list<std::string>::const_iterator itS= aL0.Name().begin();
           itS != aL0.Name().end();
           itS++
        )
        {
            aSetFin.insert(*itS);
        }
    }

    cListOfName aLON;
    std::copy(aSetFin.begin(),aSetFin.end(),back_inserter(aLON.Name()));
    MakeFileXML(aLON,aFile);
}


///==================================================
///          cMMByImNM
///==================================================

std::string cMMByImNM::NameOfType(eTypeMMByImNM aType)
{
    switch (aType)
    {
        case eTMIN_Depth : return  "Depth"  ;
        case eTMIN_Min   : return  "Min"    ;
        case eTMIN_Max   : return  "Max"    ;
        case eTMIN_Merge : return  "Merge"  ;
    }

    ELISE_ASSERT(false,"cMMByImNM::NameOfType");
    return "";
}



std::string DS2String(double aDS)
{
    if (aDS==1) return "";
    int aIDS = round_down(aDS);
    double aFrac = aDS-aIDS;

    std::string aStrDS = ToString(aDS);

    if (aFrac==0)
       aStrDS  = ToString(aIDS);


    return "DS"+ aStrDS + "_";
}


const std::string cMMByImNM::TheNamePimsFile = "PimsFile.xml";
const std::string cMMByImNM::TheNamePimsEtat = "PimsEtat.xml";

bool  cMMByImNM::StrIsPImsDIr(const std::string & aDir)
{
    return    ELISE_fp::exist_file(aDir+TheNamePimsFile)
           && ELISE_fp::exist_file(aDir+TheNamePimsEtat);
}




cMMByImNM::cMMByImNM(double aDS,const std::string & aDirGlob,const std::string & aDirLoc,const std::string & aPrefix,const std::string & aNameType,bool AddDirLoc) :
    mDS        (aDS),
    mDirGlob   (aDirGlob),
    mDirLoc    (aDirLoc),
    mPrefix    (aPrefix),
    mFullDir   (mDirGlob + mDirLoc),
    mNameFileLON ((AddDirLoc ? mDirLoc : mFullDir) + TheNamePimsFile),
    // mNameFileLON (mFullDir + TheNamePimsFile),
    mKeyFileLON (aDirGlob+ "%NKS-Set-OfFile@" + mNameFileLON),
    mNameEtat   (mFullDir+ TheNamePimsEtat),
    mNameType   (aNameType)
{
    ELISE_fp::MkDirSvp(mFullDir);
    if (ELISE_fp::exist_file(mNameEtat))
    {
       mEtats =  StdGetFromPCP(mNameEtat,EtatPims);
    }
}




void cMMByImNM::AddistofName(const cInterfChantierNameManipulateur::tSet  * aSet)
{
   ::AddistofName(mNameFileLON,aSet);
}

const std::string & cMMByImNM::KeyFileLON() const
{
   return mKeyFileLON;
}

const cEtatPims & cMMByImNM::Etat() const
{
   return mEtats;
}

void  cMMByImNM::SetOriOfEtat(const std::string & anOri)
{
   if (mEtats.NameOri().IsInit())
   {
       if (mEtats.NameOri().Val() != anOri)
       {
           std::cout << "Ori1=" << mEtats.NameOri().Val() << " Ori2=" << anOri << "\n";
           ELISE_ASSERT(false,"Multiple orientation use in PIMS");
       }
   }
   else
   {
      mEtats.NameOri().SetVal(anOri);
      MakeFileXML(mEtats,mNameEtat);
   }
}

const std::string &  cMMByImNM::GetOriOfEtat() const
{
    ELISE_ASSERT(mEtats.NameOri().IsInit(),"cMMByImNM::GetOriOfEtat");
    return mEtats.NameOri().Val();
}



const std::string PrefixMPI = "PIMs-";


std::string cMMByImNM::StdDirPims(double aDS, const std::string & aNameMatch)
{
   std::string aNameDS = DS2String(aDS);
   return  PrefixMPI  + aNameDS  + aNameMatch + "/";
}

cMMByImNM * cMMByImNM::ForGlobMerge(const std::string & aDirGlob,double aDS, const std::string & aNameMatch,bool AddDirLoc)
{
   // std::string aNameDS = DS2String(aDS);
   // std::string aDirLoc = PrefixMPI  + aNameDS  + aNameMatch + "/";
   return new cMMByImNM(aDS,aDirGlob,StdDirPims(aDS,aNameMatch),"Nuage-",aNameMatch,AddDirLoc);
}


void SelfSuppressCarDirEnd(std::string & aDir)
{
    int aL = (int)strlen(aDir.c_str());
    if (aL && (aDir[aL-1]==ELISE_CAR_DIR))
    {
        aDir = aDir.substr(0,aL-1);
    }
}
std::string SuppressCarDirEnd(const std::string & aDirOri)
{
    std::string aRes = aDirOri;
    SelfSuppressCarDirEnd(aRes);
    return aRes;
}

cMMByImNM *  cMMByImNM::FromExistingDirOrMatch(const std::string & aNameDirOri,bool Svp,double aDS,const std::string & aDir0,bool AddDirLoc)
{

// std::cout << "cMMByImNM::FromExistingDirOrMatch " << aNameDirOri << "\n"; getchar();
// ./PIMs-QuickMac/


     if (StrIsPImsDIr(aNameDirOri))
     {
         static cElRegex aRegDS(PrefixMPI + "DS(.*)_(.*)",10);
         static cElRegex aRegR1(PrefixMPI + "(.*)",10);

         std::string aFullDir=SuppressCarDirEnd(aNameDirOri);
         std::string aDirGlob,aDirLoc;
         SplitDirAndFile(aDirGlob,aDirLoc,aFullDir);

         std::string aNameMatch;
         bool Ok=false;
         if (aRegDS.Match(aDirLoc))
         {
             aDS = aRegDS.VNumKIemeExprPar(1);
             aNameMatch = aRegDS.KIemeExprPar(2);
             Ok=true;
         }
         else if (aRegR1.Match(aDirLoc))
         {
             aNameMatch = aRegR1.KIemeExprPar(1);
             Ok=true;
         }
         if (Ok)
         {
             return  cMMByImNM::ForGlobMerge(aDirGlob,aDS,aNameMatch,AddDirLoc);
         }
     }

     if (StrIsPImsDIr(StdDirPims(aDS,aNameDirOri)))
     {
        return  cMMByImNM::ForGlobMerge(aDir0,aDS,aNameDirOri,AddDirLoc);
     }


     if (! Svp)
     {
         std::cout << "For, Dir=" << aNameDirOri << " DS=" << aDS << "\n";
         ELISE_ASSERT(false,"Cannot find PIMs Directory");
     }

     return 0;
}



cMMByImNM * cMMByImNM::ForMTDMerge(const std::string & aDirGlob,const std::string & aNameIm,const std::string & aNameType)
{
   return new cMMByImNM(1.0,aDirGlob, TheDIRMergeEPI()  +   aNameIm + "/","QMNuage-",aNameType);
}


std::string cMMByImNM::NameFileGlob(eTypeMMByImNM aType,const std::string aNameIm,const std::string aExt)
{
    return mFullDir + NameFileLoc(aType,aNameIm,aExt);
}

std::string cMMByImNM::NameFileLoc(eTypeMMByImNM aType,const std::string aNameIm,const std::string aExt)
{
    return  mPrefix + NameOfType(aType) + "-"  + aNameIm  + aExt;
}




std::string cMMByImNM::NameFileLabel(eTypeMMByImNM aType,const std::string aNameIm)   {return NameFileGlob(aType,aNameIm,"_Label.tif");}
std::string cMMByImNM::NameFileMasq(eTypeMMByImNM aType,const std::string aNameIm)   {return NameFileGlob(aType,aNameIm,"_Masq.tif");}
std::string cMMByImNM::NameFileProf(eTypeMMByImNM aType,const std::string aNameIm)   {return NameFileGlob(aType,aNameIm,"_Prof.tif");}
std::string cMMByImNM::NameFileXml(eTypeMMByImNM aType,const std::string aNameIm)    {return NameFileGlob(aType,aNameIm,".xml");}
std::string cMMByImNM::NameFileEntete(eTypeMMByImNM aType,const std::string aNameIm) {return  NameFileGlob(aType,aNameIm,"");}

const std::string & cMMByImNM::FullDir() const
{
    return mFullDir;
}
const std::string & cMMByImNM::NameType() const
{
    return mNameType;
}
const std::string & cMMByImNM::DirGlob() const
{
    return mDirGlob;
}
const std::string & cMMByImNM::DirLoc() const
{
    return mDirLoc;
}

void  cMMByImNM::ModifIp(eTypeMMByImNM aType,cImage_Profondeur & anIp,const std::string & aNameIm)
{
    anIp.Image() = NameWithoutDir(NameFileProf(aType,aNameIm));
    anIp.Masq() = NameWithoutDir(NameFileMasq(aType,aNameIm));
}

/**************************************************************************/
/*                                                                        */
/*         MMEnvStatute_main                                              */
/*                                                                        */
/**************************************************************************/


int MMEnvStatute_main(int argc,char ** argv)
{
   std::string aFullName;
   std::string aPIMsDirName;
   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(aFullName,"Full Directory (Dir+Pattern)",eSAM_IsPatFile),
        LArgMain()  << EAM(aPIMsDirName,"PIMsDirName",true,"Name of PIMs directory (Statue for PIMs-Statue, Forest for PIMs-Forest", eSAM_None, ListOfVal(eNbTypeMMByP))
   );

   if (MMVisualMode) return EXIT_SUCCESS;

   std::string aDir,aNamIm;
   SplitDirAndFile(aDir,aNamIm,aFullName);

   cMMByImNM * aGlobIN = cMMByImNM::ForGlobMerge(aDir,1.0,aPIMsDirName);

   cMMByImNM * aLocIN = cMMByImNM::ForMTDMerge(aDir,aNamIm,"MTDTmp");



   //std::string aDirIn = aDir + TheDIRMergeEPI() + aNamIm + "/";


   for (int aK=0 ; aK<2 ; aK++)
   {
      bool aModeMax = (aK==0);
      eTypeMMByImNM aModIN = aModeMax ? eTMIN_Max : eTMIN_Min;
      std::string aNameMasqEnvIn =  aLocIN->NameFileMasq(aModIN,aNamIm);

      Tiff_Im aFileMasqEnvIn(aNameMasqEnvIn.c_str());


      std::string aNameProfEnvIn = aLocIN->NameFileProf(aModIN,aNamIm);
      Tiff_Im aFileProfEnvIn(aNameProfEnvIn.c_str());

      std::string aNameProfDepth = aGlobIN->NameFileProf(eTMIN_Depth,aNamIm);
      std::string aNameMasqDepth = aGlobIN->NameFileMasq(eTMIN_Depth,aNamIm);


      Tiff_Im aFileProfEnvOut = aFileProfEnvIn.Dupl(aGlobIN->NameFileProf(aModIN,aNamIm));
      Tiff_Im aFileMasqEnvOut = aFileProfEnvOut.Dupl(aGlobIN->NameFileMasq(aModIN,aNamIm));

      int aSign= (aModeMax ? 1 : -1);

      float aVDef = -1e5;
      Symb_FNum  aFMasq (Tiff_Im(aNameMasqDepth.c_str()).in(0));
      Fonc_Num  aFDepth = aSign * Tiff_Im(aNameProfDepth.c_str()).in(aSign*aVDef);

      aFDepth = aFMasq * aFDepth + (1-aFMasq) * aVDef;
      aFDepth = rect_max(aFDepth,5) + 5;
      aFDepth = Max(aFDepth,aFileMasqEnvIn.in());

      aFDepth = aFDepth * aSign;
      ELISE_COPY ( aFileProfEnvOut.all_pts(), aFDepth, aFileProfEnvOut.out());


      ELISE_COPY ( aFileMasqEnvOut.all_pts(), aFileMasqEnvIn.in()|| aFMasq, aFileMasqEnvOut.out());
/*
*/

   }

   return EXIT_SUCCESS;
}



void FiltreMasqMultiResolMMI(Im2D_REAL4 aImDepth,Im2D_U_INT1 anImInit)
{

    // Im2D_U_INT1 anImInit = Im2D_U_INT1::FromFileStd(aNameMasq);
    Pt2di aSz = anImInit.sz();
    double aCostRegul = 0.4;
    double aCostTrans = 10.0;

    //    Un filtrage basique qui supprime recursivement (par CC) les points obtenus a une resol
    //  non max et voisin du vide
    {
         Im2D_U_INT1 anImEtiq(aSz.x,aSz.y);
         ELISE_COPY(anImInit.all_pts(),Min(2,anImInit.in()),anImEtiq.out());

         ELISE_COPY(anImEtiq.border(1),0,anImEtiq.out());


         Neighbourhood aNV4=Neighbourhood::v4();
         Neigh_Rel     aNrV4 (aNV4);

         ELISE_COPY
         (
              conc
              (
                  select(select(anImEtiq.all_pts(),anImEtiq.in()==2),aNrV4.red_sum(anImEtiq.in()==0)),
                  anImEtiq.neigh_test_and_set(aNV4,2,3,256)
              ),
              3,
              Output::onul()
         );
         ELISE_COPY(select(anImEtiq.all_pts(),anImEtiq.in()==3),0,anImInit.out());
    }


    // Filtrage des irregularites par prog dyn

    {
        Im2D_U_INT1 aImMasq(aSz.x,aSz.y);
        ELISE_COPY(aImMasq.all_pts(),anImInit.in()!=0,aImMasq.out());
        ELISE_COPY(aImMasq.border(1),0,aImMasq.out());


        cParamFiltreDepthByPrgDyn aParam =  StdGetFromSI(Basic_XML_MM_File("DefFiltrPrgDyn.xml"),ParamFiltreDepthByPrgDyn);
        aParam.CostTrans().SetVal(aCostTrans);
        aParam.CostRegul().SetVal(aCostRegul);
        Im2D_Bits<1>  aNewMasq =  FiltrageDepthByProgDyn(aImDepth,aImMasq,aParam);

        // 2 est la couleur de validation
        ELISE_COPY
        (
             select(aNewMasq.all_pts(),aNewMasq.in()),
             2,
             aImMasq.out()
        );


        // supprime les ttes petites CC
        TIm2D<U_INT1,INT> aTMasq(aImMasq);
        FiltrageCardCC(false,aTMasq,2,0,100);

        //  supprime toute les CC de 1 (= avec fortes variation) voisine de 0
        Neighbourhood aNV4=Neighbourhood::v4();
        Neigh_Rel     aNrV4 (aNV4);

        ELISE_COPY
        (
           conc
           (
               select(select(aImMasq.all_pts(),aImMasq.in()==1),aNrV4.red_sum(aImMasq.in()==0)),
               aImMasq.neigh_test_and_set(aNV4,1,0,256)
           ),
           3,
           Output::onul()
        );


         ELISE_COPY(select(aImMasq.all_pts(),aImMasq.in()==0),0,anImInit.out());

    }

    // ELISE_COPY(anImInit.all_pts(),anImInit.in(),Tiff_Im(aNameMasq.c_str()).out());
}






class cAppli_Enveloppe_Main : public  cAppliWithSetImage
{
    public :
       cAppli_Enveloppe_Main(int argc,char ** argv);
       std::string NameFileCurIm(const std::string & aPref ,int aZoom)
       {
           return  mDirMergeCurIm  + aPref +  "_DeZoom" + ToString(aZoom ) + ".tif";

       }

      void DownScaleNuage(eTypeMMByImNM);
      void MakePly(const std::string &);
      void SaveResultR1(cXML_ParamNuage3DMaille  & aXMLParam,eTypeMMByImNM aType,Im2D_REAL4 aImProf,Im2D_U_INT1 anImMasq);
    private :
      int mZoom0;
      int mZoomEnd;
      int mJmp;
      std::string mNameIm;
      std::string mDirMergeCurIm;
      bool mCalledByP;
      double mScaleNuage;
      bool mShowCom;
      bool mAutoPurge;
      std::string mOut;
      cMMByImNM * mMMIN;
      cMMByImNM * mMMINS1;
      bool        mGlob;
      int         mSzW;
};



void cAppli_Enveloppe_Main::SaveResultR1(cXML_ParamNuage3DMaille  & aXMLParam,eTypeMMByImNM aType,Im2D_REAL4 aImProf,Im2D_U_INT1 anImMasq)
{
   cImage_Profondeur & anIp = aXMLParam.Image_Profondeur().Val();

   Tiff_Im::CreateFromIm(aImProf,mMMINS1->NameFileProf(aType,mNameIm));
   Tiff_Im::CreateFromFonc(mMMINS1->NameFileMasq(aType,mNameIm),anImMasq.sz(),anImMasq.in()!=0,GenIm::bits1_msbf);
   mMMINS1->ModifIp(aType,anIp,mNameIm);
   MakeFileXML(aXMLParam,mMMINS1->NameFileXml(aType,mNameIm));
}



cAppli_Enveloppe_Main::cAppli_Enveloppe_Main(int argc,char ** argv) :
   cAppliWithSetImage(argc-1,argv+1,0,TheMMByPairNameCAWSI),
   mJmp        (1),
   mCalledByP  (false),
   mScaleNuage (1),
   mShowCom    (false),
   mAutoPurge  (false),
   mOut        ("QuickMac"),
   mGlob       (true),
   mSzW        (1)
{
   std::string Masq3D;
   std::string aPat,anOri;

   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(aPat,"Full Directory (Dir+Pattern)",eSAM_IsPatFile)
                    << EAMC(anOri,"Orientation ", eSAM_IsExistDirOri)
                    << EAMC(mZoom0,"Lowest resol zoom", eSAM_IsExistDirOri)
                    << EAMC(mZoomEnd,"Largest resol zoom", eSAM_IsExistDirOri),
        LArgMain()  << EAM(Masq3D,"Masq3D",true,"3D masq for filtering")
                    << EAM(mCalledByP,"InternalCalledByP",true)
                    << EAM(mScaleNuage,"DownScale",true,"Create downscale cloud also")
                    << EAM(mShowCom,"ShowC",true,"Show command (tuning)")
                    << EAM(mAutoPurge,"AutoPurge",true,"Automatically purge unnecessary temp file (def=true)")
                    << EAM(mJmp,"Jump",true,"Will compute only image Mod Jump==0 , def=1 (all images)")
                    << EAM(mOut,"Out",true,"Target Dir in Glob Mode")
                    << EAM(mGlob,"Glob",true,"Global mode (else output in each image dir)")
                    << EAM(mSzW,"SzW",true,"Correlation Window Size (Def=1 means 3x3)")
   );


   if (! (mCalledByP))
   {
      {
          std::string aComPyr =  MM3dBinFile("MMPyram")
                                + QUOTE(aPat) + " "
                                + anOri + " "
                                + "ImSec=" +anOri;

           VoidSystem(aComPyr.c_str());
       }
       std::list<std::pair<std::string,std::string> >  aLPair = ExpandCommand(3,"InternalCalledByP=true",false,true);

       std::list<std::string>  aLCom ;
       std::list<std::string>  aNameIs ;

       for (std::list<std::pair<std::string,std::string> >::iterator itS=aLPair.begin() ; itS!=aLPair.end() ; itS++)
       {
           const std::string & aCom = itS->first;
           const std::string & aNameIm = itS->second;
           bool aDoIt = true;

           if (mJmp>1)
           {
              tSomAWSI * aSom = ImOfName(aNameIm);
              aDoIt = ((aSom->attr().mNumAccepted%mJmp)==0);
           }

           if (aDoIt)
           {
              aLCom.push_back(aCom);
           }
           else
           {
              aNameIs.push_back(aNameIm);
           }
       }

       cEl_GPAO::DoComInParal(aLCom);
       return;
   }



   ELISE_ASSERT(mVSoms.size()==1,"Only one image for cAppli_Enveloppe_Main");
   mNameIm = mVSoms[0]->attr().mIma->mNameIm;
   mDirMergeCurIm  = Dir() + TheDIRMergeEPI()  +   mNameIm + "/";

   if (mGlob)
   {
      mMMIN   =   cMMByImNM::ForGlobMerge(Dir(),mScaleNuage,mOut);
      mMMINS1 =   cMMByImNM::ForGlobMerge(Dir(),1.0,mOut);
   }
   else
   {
        mMMIN   =   cMMByImNM::ForMTDMerge(Dir(),mNameIm,"MTDTmp");
        mMMINS1 = mMMIN;
   }



   Im2D_REAL4   aEnvMax(1,1);
   Im2D_REAL4   aEnvMin(1,1);
   Im2D_U_INT1  aMasqEnv(1,1);

   Im2D_REAL4   aDepthMerg(1,1);
   Im2D_U_INT1  aMasqMerge(1,1);


   int aCpt = 0;
   for (int aZoom = mZoom0 ; aZoom >= mZoomEnd ; aZoom /= 2)
       aCpt++;

   for (int aZoom = mZoom0 ; aZoom >= mZoomEnd ; aZoom /= 2)
   {
          std::string aCom =    MM3dBinFile("MMInitialModel ")
                              + mEASF.mFullName + " "
                              + Ori()
                              + std::string(" DoPyram=false DoMatch=false  Do2Z=false   ExportEnv=true  Zoom=")
                              + ToString(aZoom)
                              + " SzW=" + ToString(mSzW)
                              ;

          if (EAMIsInit(&Masq3D)) aCom = aCom + " Masq3D=" + Masq3D;

          System(aCom);
          std::string aNameMax = NameFileCurIm("EnvMax",aZoom);
          std::string aNameMin = NameFileCurIm("EnvMin",aZoom);
          std::string aNameMasqEnv = NameFileCurIm("EnvMasq",aZoom);

          std::string aNameMerge = NameFileCurIm("Depth",aZoom);
          std::string aNameMasqMerge = NameFileCurIm("Masq",aZoom);
          if (aZoom==mZoom0)
          {
              aEnvMax = Im2D_REAL4::FromFileStd(aNameMax);
              aEnvMin = Im2D_REAL4::FromFileStd(aNameMin);
              aMasqEnv = Im2D_U_INT1::FromFileStd(aNameMasqEnv);

              aDepthMerg = Im2D_REAL4::FromFileStd(aNameMerge);
              aMasqMerge = Im2D_U_INT1::FromFileStd(aNameMasqMerge);
              ELISE_COPY(aMasqMerge.all_pts(),aMasqMerge.in()*aCpt,aMasqMerge.out());
          }
          else
          {
               ELISE_COPY
               (
                  aEnvMax.all_pts(),
                  Max(aEnvMax.in(),Tiff_Im(aNameMax.c_str()).in()),
                  aEnvMax.out()
               );
               ELISE_COPY
               (
                  aEnvMin.all_pts(),
                  Min(aEnvMin.in(),Tiff_Im(aNameMin.c_str()).in()),
                  aEnvMin.out()
               );
               ELISE_COPY
               (
                  aMasqEnv.all_pts(),
                  aMasqEnv.in() || Tiff_Im(aNameMasqEnv.c_str()).in(),
                  aMasqEnv.out()
               );


              Im2D_REAL4 aDepth= Im2D_REAL4::FromFileStd(aNameMerge);
              Im2D_U_INT1 aMasq= Im2D_U_INT1::FromFileStd(aNameMasqMerge);
              ELISE_COPY(select(aMasq.all_pts(),aMasq.in()),aDepth.in(),aDepthMerg.out());
              ELISE_COPY(select(aMasq.all_pts(),aMasq.in()),aCpt,aMasqMerge.out());
          }
          aCpt--;

          if (mShowCom)
          {
             std::cout << "COM= " << aCom << "\n";
             getchar();
          }
   }


   std::string aNameXMLIn =  mDirMergeCurIm + "NuageImProf_LeChantier_Etape_1.xml";
   cXML_ParamNuage3DMaille aXMLParam = StdGetFromSI(aNameXMLIn,XML_ParamNuage3DMaille);
   aXMLParam.Image_Profondeur().Val().Correl().SetNoInit();
   cImage_Profondeur & anIp = aXMLParam.Image_Profondeur().Val();
   anIp.Correl().SetNoInit();


   SaveResultR1(aXMLParam,eTMIN_Max,aEnvMax,aMasqEnv);
   SaveResultR1(aXMLParam,eTMIN_Min,aEnvMin,aMasqEnv);

   FiltreMasqMultiResolMMI(aDepthMerg,aMasqMerge);
   SaveResultR1(aXMLParam,eTMIN_Depth,aDepthMerg,aMasqMerge);

   // A revoir mais pour l'instant si non Glob ca se marche sur le pied
   if ((mScaleNuage!=1.0) && mGlob)
   {
      DownScaleNuage(eTMIN_Max);
      DownScaleNuage(eTMIN_Min);
      DownScaleNuage(eTMIN_Depth);
   }



   if (mAutoPurge)
   {

       std::string aDirMatch  = Dir() + "Masq-TieP-" + mNameIm  + "/";
       ELISE_fp::PurgeDir(aDirMatch,true);

       if (mGlob)
       {
           ELISE_fp::PurgeDir(mDirMergeCurIm,true);
       }
       else
       {
           std::string aVS[TheNbPref] ={"EnvMax","EnvMin","EnvMasq","Depth","Masq"};

           for (int aZoom = mZoom0 ; aZoom >= mZoomEnd ; aZoom /= 2)
           {
              for (int aK=0 ; aK<5 ; aK++)
              {
                   ELISE_fp::RmFile(Dir()+NameFileCurIm(aVS[aK],aZoom));
              }
           }
       }
   }
}

void  cAppli_Enveloppe_Main::MakePly(const std::string & aNN)
{
    std::string aCom =  MM3dBinFile("Nuage2Ply") + "  " + aNN  ;
    System(aCom);
}

void cAppli_Enveloppe_Main::DownScaleNuage(eTypeMMByImNM aType)
{

    std::string aCom =     MM3dBinFile("ScaleNuage")
                        + " " + mMMINS1->NameFileXml(aType,mNameIm)
                        + " " + mMMIN->NameFileEntete(aType,mNameIm)
                        + " " + ToString(mScaleNuage)
                        + " InDirLoc=false"
                      ;

    System(aCom);
}

int MMEnveloppe_Main(int argc,char ** argv)
{
   cAppli_Enveloppe_Main(argc,argv);
   return EXIT_SUCCESS;
}



  //=========================================

int MMInitialModel_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv);

    std::string  aDir,aPat,aFullDir;
    std::string  AeroIn;
    std::string  ImSec;
    bool         Visu = false;
    bool         ExportEnv = false;
    bool         DoPly = false;
    bool         DoMatch = true;

    int aZoom = 8;
    bool aDo2Z = true;
    double aReducePly=3.0;
    std::string aMasq3D;
    bool aDoPyram= true;
    int  aSzW=1;

    ElInitArgMain
    (
    argc,argv,
    LArgMain()  << EAMC(aFullDir,"Dir + Pattern", eSAM_IsPatFile)
                    << EAMC(AeroIn,"Orientation", eSAM_IsExistDirOri),
    LArgMain()
                    << EAM(Visu,"Visu",true,"Interactif Visualization (tuning purpose, program will stop at breakpoint)", eSAM_IsBool)
                    << EAM(DoPly,"DoPly",true,"Generate ply ,for tuning purpose (Def=false)", eSAM_IsBool)
                    << EAM(aZoom,"Zoom",true,"Zoom of computed models (Def=8)")
                    << EAM(aReducePly,"ReduceExp",true,"Down scaling of cloud , XML and ply (Def=3)")
                    << EAM(aDo2Z,"Do2Z",true,"Excute a first step at 2*Zoom (Def=true)", eSAM_IsBool)
                    << EAM(DoMatch,"DoMatch",true,"Do \"classical\" MicMac at end (Def=true)", eSAM_IsBool)
                    << EAM(aMasq3D,"Masq3D",true,"3D masq when exist (Def=true)", eSAM_IsBool)
                    << EAM(ExportEnv,"ExportEnv",true,"Export Max Min surfaces (Def=false)", eSAM_IsBool)
                    << EAM(aDoPyram,"DoPyram",true,"Do Pyram (set false when //izing)  ", eSAM_IsBool)
                    << EAM(aSzW,"SzW",true,"Correlation Window Size (Def=1 means 3x3)")
    );

    if (MMVisualMode) return EXIT_SUCCESS;

    #if (ELISE_windows)
        replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
    #endif
    SplitDirAndFile(aDir,aPat,aFullDir);
    StdCorrecNameOrient(AeroIn,aDir);

    cInterfChantierNameManipulateur * aICNM =  cInterfChantierNameManipulateur::BasicAlloc(aDir);

    if (! EAMIsInit(&ImSec))
       ImSec = AeroIn;

    // Genere les pryramides pour que le paral ne s'ecrase pas les 1 les autres
    if (aDoPyram)
    {
         std::string aComPyr =  MM3dBinFile("MMPyram")
                                + QUOTE(aFullDir) + " "
                                + AeroIn + " "
                                + "ImSec=" +ImSec;

         VoidSystem(aComPyr.c_str());
    }

    const cInterfChantierNameManipulateur::tSet * aSetIm = aICNM->Get(aPat);

    std::list<std::string> aLCom;

    for (int aKIm=0 ; aKIm<int(aSetIm->size()) ; aKIm++)
    {
          std::string aCom =   MM3dBinFile("MICMAC")
                              //  + XML_MM_File("MM-ModelInitial.xml")
                              + XML_MM_File("MM-TieP.xml")
                              + std::string(" WorkDir=") +aDir +  std::string(" ")
                              + std::string(" +Im1=") + QUOTE((*aSetIm)[aKIm]) + std::string(" ")
                              + std::string(" +Ori=-") + AeroIn
                              + std::string(" +ImSec=-") + ImSec
                              + " +DoPly=" + ToString(DoPly) + " "
                              + " +DoMatch=" + ToString(DoMatch) + " "
                              + " +SzW=" + ToString(aSzW) + " "
                    ;

          if (ExportEnv)
              aCom = aCom + " +ExportEnv=true";

          if (Visu)
              aCom = aCom + " +Visu=" + ToString(Visu) + " ";

          if (EAMIsInit(&aZoom))
             aCom = aCom + " +Zoom=" + ToString(aZoom) + " ";

          if (EAMIsInit(&aDo2Z))
             aCom = aCom + " +Do2Z=" + ToString(aDo2Z) + " ";

          if (EAMIsInit(&aReducePly))
             aCom = aCom + " +ReduceExp=" + ToString(aReducePly) + " ";
          if (EAMIsInit(&aMasq3D))
             aCom = aCom + " +UseMasq3D=true +FileMasq3D=" + aMasq3D + " ";
          std::cout << "Com = " << aCom << "\n";
          aLCom.push_back(aCom);
  }

  cEl_GPAO::DoComInParal(aLCom);
  // cEl_GPAO::DoComInParal(aLCom,"MkMMInit");
 // int aRes = system_call(aCom.c_str());

   // int i; DoNothingButRemoveWarningUnused(i);

   return EXIT_SUCCESS;
}





/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
