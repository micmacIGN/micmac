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

#include "NewOri.h"
#include "../TiepTri/MultTieP.h"

/*
int TestNewOriImage_main(int argc,char ** argv)
{
   std::string aNameOri,aNameI1,aNameI2;
   ElInitArgMain
   (
        argc,argv,
        LArgMain() <<  EAMC(aNameI1,"Name First Image")
                   <<  EAMC(aNameI2,"Name Second Image"),
        LArgMain() << EAM(aNameOri,"Ori",true,"Orientation ")
   );


    cNewO_NameManager aNM("./",aNameOri,"dat");

    CamStenope * aC1 = aNM.CamOfName(aNameI1);
    CamStenope * aC2 = aNM.CamOfName(aNameI2);

    ElPackHomologue aLH = aNM.PackOfName(aNameI1,aNameI2);

    std::cout << "FFF " << aC1->Focale() << " " << aC2->Focale() << " NBh : " << aLH.size() << "\n";

    return EXIT_SUCCESS;
}
*/

///Export the graph to G2O format for testing in ceres
class cAppliNOExport;
class Constraint;
class Pose3d;
class RotMat;


//read SfmInit input
class cAppliImportSfmInit;

//namespace ceresTestER
//{
class RotMat 
{
    public :
        RotMat(const Pt3dr &aI,const Pt3dr &aJ,const Pt3dr &aK) : 
            mI(aI), 
            mJ(aJ),
            mK(aK) {};
        
        Pt3dr & I(){return mI;}
        Pt3dr & J(){return mJ;}
        Pt3dr & K(){return mK;}

    private :
        Pt3dr   mI;
        Pt3dr   mJ;
        Pt3dr   mK;
};

class Pose3d
{
    public :
        Pose3d(const Pt3dr &aP,
               const Pt3dr &aI,const Pt3dr &aJ,const Pt3dr &aK,
               const int aId) : 
            mP(aP),
            mQ(aI,aJ,aK),
            mId(aId) {};

        int    & Id(){return mId;};
        Pt3dr  & P(){return mP;}
        RotMat & R(){return mQ;}

        static std::string name() {return "VERTEX_SE3:QUAT";};
    
    private : 
        Pt3dr              mP;
        RotMat             mQ;
        int                mId;


};

class Constraint
{
    public :
        Constraint(const int & aI0,const int & aI1,
                   const Pose3d aRel,
                   const Pt3dr  aPdsT,
                   const Pt3dr  aPdsR) : 
            mI0(aI0),
            mI1(aI1),
            mRel(aRel),
            mPdsR(aPdsR),
            mPdsT(aPdsT) {};
            
        int    &  I0(){return mI0;};
        int    &  I1(){return mI1;};
        Pose3d &  Pose(){return mRel;};
        Pt3dr  &  PdsR(){return mPdsR;}
        Pt3dr  &  PdsT(){return mPdsT;}

        static std::string name() {return "EDGE_SE3:QUAT";}
            
    private :
        int      mI0,mI1; 
        Pose3d   mRel;
        Pt3dr    mPdsR;
        Pt3dr    mPdsT;
        
};

class cAppliNOExport : public cCommonMartiniAppli
{
    public : 
        cAppliNOExport(int argc,char ** argv);

    private :
        bool NOSave(const std::map<std::string,Pose3d *> aMP,
                    const std::vector<Constraint*> aCVec,
                    const std::string & aName );
        void NOSaveConstraint(std::fstream* aFile, Constraint* aC);
        void NOSaveNoed(std::fstream* aFile,Pose3d* aMP);
};


///pose X Y Z rot_I_x rot_I_y rot_I_z rot_J_x rot_J_y rot_J_z rot_K_x rot_K_y rot_K_z
void cAppliNOExport::NOSaveNoed(std::fstream* aFile,Pose3d* aMP)
{
    *aFile << aMP->name().c_str() << " " << aMP->Id() << " " << aMP->P().x << " " << aMP->P().y << " " << aMP->P().z << 
       " " << aMP->R().I().x << " " << aMP->R().I().y << " " << aMP->R().I().z << 
       " " << aMP->R().J().x << " " << aMP->R().J().y << " " << aMP->R().J().z << 
       " " << aMP->R().K().x << " " << aMP->R().K().y << " " << aMP->R().K().z << "\n";
};

///ID_a ID_b x_ab y_ab z_ab q_x_ab q_y_ab q_z_ab q_w_ab I_11 I_12 I_13 ... I_16 I_22 I_23 ... I_26 ... I_66 
void cAppliNOExport::NOSaveConstraint(std::fstream* aFile, Constraint* aC)
{
    *aFile << aC->name().c_str() << " " << aC->I0() << " " << aC->I1() << " " 
           << aC->Pose().P().x << " " << aC->Pose().P().y << " " << aC->Pose().P().z << " "
           << aC->Pose().R().I().x << " " << aC->Pose().R().I().y << " " << aC->Pose().R().I().z << " "
           << aC->Pose().R().J().x << " " << aC->Pose().R().J().y << " " << aC->Pose().R().J().z << " "
           << aC->Pose().R().K().x << " " << aC->Pose().R().K().y << " " << aC->Pose().R().K().z << " "
           << aC->PdsR().x << " 0 0 0 0 0 " 
           << "0 " << aC->PdsR().y << " 0 0 0 0 "
           << "0 0 " << aC->PdsR().z << " 0 0 0 "
           << "0 0 0 " << aC->PdsT().x << " 0 0 " 
           << "0 0 0 0 " << aC->PdsT().y << " 0 "
           << "0 0 0 0 0 " << aC->PdsT().z << "\n";

            
};

bool cAppliNOExport::NOSave(const std::map<std::string,Pose3d *> aMP,
          const std::vector<Constraint*> aCVec,
          const std::string & aName )
{
   std::fstream aOut;
   aOut.open(aName.c_str(), std::istream::out); 

   if (!aOut) 
   {
        ELISE_ASSERT
        (
                false,
                "NewOriImage2G2O_main save; can't open file"
        );
		aOut.close();
        return false;
   }

   for(auto aK : aMP)
   {
       NOSaveNoed(&aOut,aK.second);
   }

   for(auto aK : aCVec)
   {
       NOSaveConstraint(&aOut,aK);
   }

   aOut.close();

   return true;


};




cAppliNOExport::cAppliNOExport(int argc,char ** argv) :
    cCommonMartiniAppli ()
{

    std::string aPat,aDir;
    std::string aOri="Martini";
    std::string aName="triplets_g2o.txt";

    cInterfChantierNameManipulateur * aICNM;
    std::list<std::string> aLFile;

    ElInitArgMain
    (
        argc,argv,
        LArgMain() << EAMC(aPat,"Pattern of images", eSAM_IsExistFile),
        LArgMain() << EAM(aOri,"Ori",true,"Initial absolute ori; Def=[Ori-Martini]")
                   << EAM(aName,"Out",true,"Output file name")
    );

    #if (ELISE_windows)
        replace( aPat.begin(), aPat.end(), '\\', '/' );
    #endif

    SplitDirAndFile(aDir,aPat,aPat);
    StdCorrecNameOrient(aOri,aDir);

    /// get map of initial orientations
    aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    aOri= aICNM->StdKeyOrient(aOri);
    aLFile =  aICNM->StdGetListOfFile(aPat,1);

    std::map<std::string,Pose3d *> aMP;

    int aNCP=0;
    for( auto aL : aLFile )
    {

        std::string aNF = aICNM->Dir() + aICNM->Assoc1To1(aOri,aL,true);
        Pt3dr aC = StdGetObjFromFile<Pt3dr>
                (
                    aNF,
                    StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                    "Centre",
                    "Pt3dr"
                );        

        cOrientationConique * aCO = OptionalGetObjFromFile_WithLC<cOrientationConique>
                                 (
                                       0,0,
                                       aNF,
                                       StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                       "OrientationConique",
                                       "OrientationConique"
                                 );
        cRotationVect aRV   = aCO->Externe().ParamRotation();
        
        std::cout << "+P(" << aL << ")=" << aNCP << "\n";

        aMP[aL] = new Pose3d(aC,
                         aRV.CodageMatr().Val().L1(),
                         aRV.CodageMatr().Val().L2(),
                         aRV.CodageMatr().Val().L3(),
                         aNCP++);
    
    }        
    std::cout << "No de noeds=" << aNCP << "\n";


    ///triplets dir manager
    cNewO_NameManager *  aNM = NM(aDir);
    std::string aNameLTriplets = aNM->NameTopoTriplet(true);
    cXml_TopoTriplet  aLT = StdGetFromSI(aNameLTriplets,Xml_TopoTriplet);
    

    ///get vector of constraints
    int aNCC=0;
    std::vector<Constraint*> aCVec;
    
    for (auto a3 : aLT.Triplets())
    {
        if (DicBoolFind(aMP,a3.Name1()) && DicBoolFind(aMP,a3.Name2()) && DicBoolFind(aMP,a3.Name3()))
        {
            std::string  aName3R = aNM->NameOriOptimTriplet(true,a3.Name1(),a3.Name2(),a3.Name3());
            cXml_Ori3ImInit aXml3Ori = StdGetFromSI(aName3R,Xml_Ori3ImInit);
            
            //find id inside aMP
            Pose3d *aPA1 = aMP[a3.Name1()];
            Pose3d *aPA2 = aMP[a3.Name2()];
            Pose3d *aPA3 = aMP[a3.Name3()];
     
    

            ///1-2
            ElRotation3D aP12 = Xml2El(aXml3Ori.Ori2On1());      
            
            Pt3dr aI,aJ,aK;
            aP12.Mat().GetCol(0,aI);
            aP12.Mat().GetCol(1,aJ);
            aP12.Mat().GetCol(2,aK);
     
            
            aCVec.push_back(new Constraint( aPA1->Id(),aPA2->Id(), 
                                            Pose3d(aP12.tr(),aI,aJ,aK,aNCC++),
                                            Pt3dr(1,1,1),
                                            Pt3dr(1,1,1)
                                            ));
       
            std::cout << "C=(" << aPA1->Id() << "," << aPA2->Id() << ")="  << a3.Name1() << "-" << a3.Name2() << "\n"; 
            ///1-3
            ElRotation3D aP13 = Xml2El(aXml3Ori.Ori3On1());
         
            aP13.Mat().GetCol(0,aI);
            aP13.Mat().GetCol(1,aJ);
            aP13.Mat().GetCol(2,aK);
         
            aCVec.push_back(new Constraint( aPA1->Id(),aPA3->Id(), 
                                            Pose3d(aP13.tr(),aI,aJ,aK,aNCC++),
                                            Pt3dr(1,1,1),
                                            Pt3dr(1,1,1)));
            
            std::cout << "C=(" << aPA1->Id() << "," << aPA3->Id() << ")=" << a3.Name1() << "-" << a3.Name3() << "\n"; 
        }
    }
    std::cout << "No de contraints=" << aNCC << "\n";



   
    NOSave(aMP,aCVec,aName);
            
}

int CPP_NewOriImage2G2O_main(int argc,char ** argv)
{
    cAppliNOExport aAppli(argc,argv);
    return EXIT_SUCCESS;

}
//}

typedef std::map<int,Pt2dr > tKeyPt;
struct CamSfmInit
{
    std::string mName;
    Pt2dr  PP;
    double F;

    CamSfmInit(std::string n="", Pt2dr pp=Pt2dr(0,0), double f=0) : 
            mName(n),
            PP(pp),
            F(f){}

};


class cAppliImportSfmInit 
{
    public:
        cAppliImportSfmInit(int argc,char ** argv);

        bool Read();

		bool Save();

    private :
		/* Read */
        bool ReadCC();
        bool ReadCoords();
        bool ReadCoordsOneCam(FILE *fptr,int &aCamId, tKeyPt &aKeyPts);
        void ConvMMConv(Pt2dr &aPt);

        bool ReadEdges();
		bool ReadSolution();

		std::string SolutionOriName();

        void SaveCalib();
        void SaveOC(ElMatrix<double>& aR, Pt3dr& aTr, int& aCamId);
        void ShowImgs();
        void DoListOfName();
        std::string GetListOfName();

		bool WriteMMOri2Txt();


		/* Save */
		bool SaveCC();
		bool SaveCoords();
 
        template <typename T>
        void FileReadOK(FILE *fptr, const char *format, T *value);

        cInterfChantierNameManipulateur * mICNM;
        bool DoCalib;
        bool DoImags;



        std::string mDir;
        std::string mCCListAllFile;
        std::string mCCFile;
        std::string mCoordsFile;
        std::string mTracksFile;
        std::string mEGFile;
        std::string mRotSolFile;
        std::string mTrSolFile;
        std::string mMMOriDir;
        std::string mSH;
        std::string mPost;

		std::string mPat;
		std::list<std::string> mLFile;

        std::string CalibDir;

        std::map<int,CamSfmInit *>       mCC;//int follows the SfmInit indexing
        std::map<int,int>                mSfm2MM_ID;//mapping of the sfm indexes to MM indexes; the latter must follow the image order in mCCVec
        std::vector<std::string> *       mCCVec;//Vec to initialise the merge structure
        std::map<int,tKeyPt >            mC2KPts; //map containing a keypoint set per image; the int is the image id as in mCC

        //cVirtInterf_NewO_NameManager *   mVNM;
};

template <typename T>
void cAppliImportSfmInit::FileReadOK(FILE *fptr, const char *format, T *value)
{
    int OK = fscanf(fptr, format, value);
    if (OK != 1)
        ELISE_ASSERT(false, "cAppliImportSfmInit::FileReadOK")

}

std::string cAppliImportSfmInit::GetListOfName()
{
    return "ListOfFiles.xml";
}

void cAppliImportSfmInit::DoListOfName()
{

    if (mCCVec)
    {
        cListOfName aLON;
        for (int aI=0; aI<int(mCCVec->size()); aI++)
        {
            aLON.Name().push_back(mCCVec->at(aI));
        }

        MakeFileXML(aLON,GetListOfName());
    }
}

void cAppliImportSfmInit::ShowImgs()
{
    if (mCCVec)
    {
        for (int aI=0; aI<int(mCCVec->size()); aI++) 
        {
            std::cout << " " << mCCVec->at(aI) << "\n";
        }

    }
}

bool cAppliImportSfmInit::ReadCC()
{

    /* Read the list */
    std::vector<std::string> aNameList;
    {
        ELISE_fp aFIn(mCCListAllFile.c_str(),ELISE_fp::READ);
        char * aLine;
 

        while ((aLine = aFIn.std_fgets()))
        {
 
            char aName[50];
            int         aNull=0;
            double      aF=0;
 
            int aNb=sscanf(aLine,"%s %i %lf", aName, &aNull, &aF);

 
            ELISE_ASSERT((aNb==3) || (aNb==1),"Could not read 3 or 1 values");
 
            aNameList.push_back(NameWithoutDir(aName));
 
        }
        aFIn.close();
    }
       
    /* Read the cc and associate with the list */
    int aMMID=0;
    {
        ELISE_fp aFIn(mCCFile.c_str(),ELISE_fp::READ);
        char * aLine;

        while ((aLine = aFIn.std_fgets()))
        {

            int         aIdx=0;

            int aNb=sscanf(aLine,"%i", &aIdx);
            ELISE_ASSERT((aNb==1),"Could not read the id");

            mCC[aIdx] = new CamSfmInit (aNameList.at(aIdx),Pt2dr(0,0),0); 
            mCCVec->push_back(aNameList.at(aIdx)); 
            mSfm2MM_ID[aIdx] = aMMID;
            aMMID++;
        }
        aFIn.close();
   
    }

    return true;
}

/* MM:      origin at (0,0)
   SfmInit: origin at (1,1) */
void cAppliImportSfmInit::ConvMMConv(Pt2dr &aPt)
{
    aPt.x = aPt.x -1;
    aPt.y = aPt.y -1;
}

/* Read keypts per camera and
    update io in mCC  */

bool cAppliImportSfmInit::ReadCoordsOneCam(FILE *fptr,int &aCamId, tKeyPt &aKeyPts)
{
    
    char line[50];
    for (int aIt=0; aIt<2; aIt++)
    {
        FileReadOK(fptr, "%s", line);
    }
    FileReadOK(fptr, "%i,", &aCamId);

    for (int aIt=0; aIt<5; aIt++)
    {
        FileReadOK(fptr, "%s", line);
    }
    int aNbKey;
    FileReadOK(fptr, "%i", &aNbKey);
    //std::cout << "Nb keys: " << aNbKey << "\n";

    
    for (int aIt=0; aIt<3; aIt++)   
    {
        FileReadOK(fptr, "%s", line);
    }


    FileReadOK(fptr, "%lf,", &(mCC[aCamId]->PP.x));
    

    for (int aIt=0; aIt<2; aIt++)
    {
        FileReadOK(fptr, "%s", line);
    }
    FileReadOK(fptr, "%lf,", &(mCC[aCamId]->PP.y));
//    std::cout << "PP: " << mCC[aCamId]->PP << "\n";

    for (int aIt=0; aIt<2; aIt++)
    {
        FileReadOK(fptr, "%s", line);
    }
    FileReadOK(fptr, "%lf", &(mCC[aCamId]->F));
//    std::cout << "F: " << mCC[aCamId]->F << "\n";


    int   aPtId;
    Pt2dr aPt;
    Pt2di aIgnr;
    Pt3di aRGB;
    for (int aK=0; aK<aNbKey; aK++)
    {
        int OK = std::fscanf(fptr,"%i %lf %lf %i %i %i %i %i\n",&aPtId,&aPt.x,&aPt.y,&aIgnr.x,&aIgnr.y,&aRGB.x,&aRGB.y,&aRGB.z);
        if (OK)
        {
//          std::cout << aPtId << " " << aPt << " " << aRGB << "\n";
            ConvMMConv(aPt);
            aKeyPts[aPtId] = aPt;
        }
        else 
        {
            std::cout << "cAppliImportSfmInit::ReadCoordsOneCam could not read a line" << "\n";
            return EXIT_FAILURE;
        }

    }


    return EXIT_SUCCESS;
}

/* 1/ Read keypts per camera and update io
   2/ Read tracks and create Homol

   Decoding coords, eg:
   2 0 0 1 0 
    - a tie-pts visible in 2 images
    - image id 0, point id 0
    - image id 1, point id 0 */

bool cAppliImportSfmInit::ReadCoords()
{

    /* Keypts per camera */
    {
        FILE* fptr = fopen(mCoordsFile.c_str(), "r");
        if (fptr == NULL) {
          return false;
        };
 
 
        while (!std::feof(fptr)  && !ferror(fptr))
        {

            tKeyPt  aKPtsPerCam;
            int      aCamIdx;
            ReadCoordsOneCam(fptr,aCamIdx,aKPtsPerCam);
 
            mC2KPts[aCamIdx] = aKPtsPerCam;
 
            /*     tKeyPt  ttt = mC2KPts[aCamIdx]; //aCamIdx camera
            Pt2dr aaa = (ttt)[0];//first keypt of the aCamIdx camera
            std::cout << "mC2KPts[0]: " << aaa << " " << " " << " "  << "\n"; //(*ttt)[0] */
 
        }

        fclose(fptr);
    }

    { 
        /* Tracks and homol */
        FILE* fptr = fopen(mTracksFile.c_str(), "r");
        if (fptr == NULL) {
          return false;
        };
 
        
        int NbTrk;
        FileReadOK(fptr, "%i", &NbTrk);
//        std::cout << "&NbTrk " << NbTrk << "\n";
 
        cSetTiePMul * aMulHomol = new cSetTiePMul(0, mCCVec);
        std::vector<std::vector<int>>   VNumCams;
        std::vector<std::vector<Pt2dr>> VPtsCams;
 
 
        /* Iterate over tracks */
        for (int aT=0; aT<NbTrk; aT++)
        {
            std::vector<int>   VNum;
            std::vector<Pt2dr> VPts;
 
            int aTrkLen;
            FileReadOK(fptr, "%i", &aTrkLen);
//            std::cout << "&aTrkLen " << aTrkLen << "\n";
 
            /* Colect the track aT */
            for (int aK=0; aK<aTrkLen; aK++)
            {
                int aCamID;
                int aPtID;
 
                FileReadOK(fptr, "%i", &aCamID);
                FileReadOK(fptr, "%i", &aPtID);
 
                VNum.push_back(mSfm2MM_ID[aCamID]);
 
                //std::cout << "(mC2KPts[aCamID])[aPtID] " << aCamID << " " << mSfm2MM_ID[aCamID] << " " << aPtID << " " << (mC2KPts[aCamID])[aPtID] << "\n";
                VPts.push_back((mC2KPts[aCamID])[aPtID]);
 
            }
            VNumCams.push_back(VNum);
            VPtsCams.push_back(VPts);
        }
        fclose(fptr);
      
        for (uint aK=0; aK<VNumCams.size(); aK++)
        {
            vector<float> vAttr;
            aMulHomol->AddPts(VNumCams[aK], VPtsCams[aK],vAttr);
        }

 
        std::string aSave = cSetTiePMul::StdName(mICNM,mSH,"-SfmInit",false); 
        aMulHomol->Save(aSave);
    }


    return true;
}

void cAppliImportSfmInit::SaveCalib()
{
	int aNb = mCC.size();
	int aC=1;
    for (auto aIm : mCC)
    {
        std::cout << "aIm " << " " << aIm.second->mName << " " << aC << " / " << aNb << "\n";
        //std::cout << "aIm " << mICNM->StdNameCalib(CalibDir,aIm.second->mName)  << "\n";

        cCalibrationInternConique aCIO = StdGetObjFromFile<cCalibrationInternConique>
                (
                    Basic_XML_MM_File("Template-Calib-Basic.xml"),
                    StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                    "CalibrationInternConique",
                    "CalibrationInternConique"
                );

        aCIO.PP()   = aIm.second->PP ;
        aCIO.F()    = aIm.second->F ;
        aCIO.SzIm() = Pt2di(2*aIm.second->PP.x,2*aIm.second->PP.y); //SfmInit convention
        aCIO.CalibDistortion()[0].ModRad().Val().CDist() = Pt2dr(0,0);

        MakeFileXML(aCIO,mICNM->StdNameCalib(CalibDir,aIm.second->mName));
		aC++;
    }



}

bool cAppliImportSfmInit::ReadEdges()
{
    /* -read edge
       -associate with cam (io)
       -read R and t
       -save to NewTmp ...
           + Xml_Ori2Im
                - im name, 
                - calib 
                - NbPts
                - Foc1 Foc2 FocMoy
                - Box?

std::map<int,CamSfmInit *>       mCC;
 
*/
    FILE* fptr = fopen(mEGFile.c_str(), "r");
    if (fptr == NULL) {
      return false;
    };


    while (!std::feof(fptr)  && !ferror(fptr))
    {
        Pt2di aE;
        Pt3dr        aTij;
        ElMatrix<double> aRij(3,3,1.0);

        bool OK = std::fscanf(fptr, "%i %i %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", 
                          &aE.x, &aE.y, 
                          &aRij(0,0), &aRij(0,1), &aRij(0,2),
                          &aRij(1,0), &aRij(1,1), &aRij(1,2),
                          &aRij(2,0), &aRij(2,1), &aRij(2,2),
                          &aTij.x, &aTij.y, &aTij.z);

        std::cout << aRij(0,0) << " " << aRij(0,1) << " " << aRij(0,2) << "\n"
                  << aRij(1,0) << " " << aRij(1,1) << " " << aRij(1,2) << "\n"
                  << aRij(2,0) << " " << aRij(2,1) << " " << aRij(2,2) << "\n";
        std::cout << "T " << aTij << " E " << aE << " " << OK << "\n";

        getchar();


    }
//to do 
    fclose(fptr);

    return EXIT_SUCCESS;
}

std::string cAppliImportSfmInit::SolutionOriName()
{
	return ("SfmInit/");
}

void cAppliImportSfmInit::SaveOC(ElMatrix<double>& aR, Pt3dr& aTr, int& aCamId)
{

	
	std::string aImN = mCC[aCamId]->mName;

	/* internal calibration */
	std::string aCalibName = mICNM->StdNameCalib(CalibDir,aImN);


	/* external */
	std::string aFileExterne = "Ori-" + SolutionOriName() + "Orientation-" + aImN + ".xml";
	std::cout << aFileExterne <<  " " << aCalibName << "\n";

	cOrientationExterneRigide aExtern;

	//conv MM is Camera2Monde and Bundler Monde2Camera
    ElMatrix<double> aRotMM = aR.transpose();
    Pt3dr            aTrMM  = aTr;//- (aR*aTr);

	std::cout << "aTrMM=" << aTrMM << "\n";

	/* necessary? */
    if (! (isnan(aTrMM.x) || isnan(aTrMM.y) || isnan(aTrMM.z)))
    {
        aExtern.Centre() = aTrMM;
        aExtern.IncCentre() = Pt3dr(1,1,1);


        cTypeCodageMatr aTCRot;
        aTCRot.L1() = Pt3dr(aRotMM(0,0),-aRotMM(0,1),-aRotMM(0,2));     //  1  0            0            conv Bundler: Z points away from the scene
        aTCRot.L2() = Pt3dr(aRotMM(1,0),-aRotMM(1,1),-aRotMM(1,2));   // 0 cos teta=-1   -sin teta=0
        aTCRot.L3() = Pt3dr(aRotMM(2,0),-aRotMM(2,1),-aRotMM(2,2));   // 0 sin teta=0   cos teta=-1
        aTCRot.TrueRot() = true;

        cRotationVect aRV;
        aRV.CodageMatr() = aTCRot;
        aExtern.ParamRotation() = aRV;

		cOrientationConique aOC;
        aOC.ConvOri().KnownConv().SetVal(eConvApero_DistM2C);
        aOC.Externe() = aExtern;
        aOC.FileInterne().SetVal(aCalibName);

        MakeFileXML(aOC,aFileExterne);
    }


}

bool cAppliImportSfmInit::ReadSolution()
{

	FILE* fptrR = fopen(mRotSolFile.c_str(), "r");
    if (fptrR == NULL) {
      return false;
    };


	FILE* fptrT = fopen(mTrSolFile.c_str(), "r");
    if (fptrT == NULL) {
      return false;
    };




    for (int aK=0 ; aK<int(mCC.size()); aK++)
    {
		//instrad of temporary vars I could keep it in a class
		Pt3dr aTr;
		ElMatrix<double> aR(3,3,1.0);
		int aIdR;
		int aIdT;

		//read rotation
		if (!std::feof(fptrR)  && !ferror(fptrR))
		{

        	bool OK = std::fscanf(fptrR, "%i %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                          &aIdR,
                          &aR(0,0), &aR(0,1), &aR(0,2),
                          &aR(1,0), &aR(1,1), &aR(1,2),
                          &aR(2,0), &aR(2,1), &aR(2,2));

			if (! OK)
				return EXIT_FAILURE;

		}
		else
			return EXIT_FAILURE;

		//read translation
		if (!std::feof(fptrT)  && !ferror(fptrT))
        {
			bool OK = std::fscanf(fptrT,"%i %lf %lf %lf",
							      &aIdT,
								  &aTr.x,&aTr.y,&aTr.z);

			if (! OK)
				return EXIT_FAILURE;

		}
		else
            return EXIT_FAILURE;

		//if ids agree
		if (aIdR==aIdT)
		{

			SaveOC (aR,aTr,aIdR);
		}
	}

    fclose(fptrR);
    fclose(fptrT);


	return EXIT_SUCCESS;
}

bool cAppliImportSfmInit::WriteMMOri2Txt()
{

	std::string aNameSave = "micmac_centers.txt";
	
	std::fstream aOut;
    aOut.open(aNameSave.c_str(), std::istream::out);

	for (auto aIm : mCC)
    {
        std::cout << "aIm " << " " << aIm.first << " " << aIm.second->mName << " " << "\n"; 
     	std::string aNF = mICNM->Dir() + mICNM->Assoc1To1(mMMOriDir,aIm.second->mName,true);

		if (ELISE_fp::exist_file(aNF))
		{

			Pt3dr aC = StdGetObjFromFile<Pt3dr>
                	(
                    	aNF,
                    	StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                    	"Centre",
                    	"Pt3dr"
                	);

			aOut << aIm.first << " " << aC.x << " " << aC.y << " " << aC.z << "\n";

		}

	}	

	aOut.close();
	
	return true;
}

bool cAppliImportSfmInit::Read()
{
    if (ReadCC())
        std::cout << "[ImportSfmInit] Read cc.txt, done!" << "\n";
    else 
        return false;



    if (ReadCoords())     
        std::cout << "[ImportSfmInit] Read coords.txt and tracks.txt, done!" << "\n";
    else 
        return false;



    if (DoImags)
    {
        DoListOfName();
        ShowImgs();        
        std::cout << "[ImportSfmInit] Image list saved to: " << GetListOfName() << "\n";
    }


    if (CalibDir != "")
    {
		if ((mRotSolFile != "") && (mTrSolFile != ""))
			CalibDir = SolutionOriName();

        SaveCalib();
        std::cout << "[ImportSfmInit] Image calibrations saved to: " << "Ori-" + CalibDir << "\n";
    }


    if (mEGFile != "")
    {
        if (ReadEdges())
            std::cout << "[ImportSfmInit] Edges saved to: " << "\n";
    }

	if ( (mRotSolFile != "") && (mTrSolFile != ""))
	{
		if (CalibDir == "")
		{
			CalibDir = SolutionOriName();
		//	SaveCalib();

        	std::cout << "[ImportSfmInit] Image calibrations saved to: " << CalibDir << "\n";
		}

		if (ReadSolution())
            std::cout << "[ImportSfmInit] rotation / translation solution done. " << "\n";
	}

	if (mMMOriDir != "")
	{
		if (WriteMMOri2Txt())
		{
			std::cout << "[ImportSfmInit] Orientations saved to: " << "\n";
			return true;
		}
		else
			return false;
	}	

    return true;
}

/****************************** SAVE *******************************/

bool cAppliImportSfmInit::SaveCoords()
{
	cSetTiePMul * aSetPM = new cSetTiePMul(0);


	std::string aPMulFile;
	if (ELISE_fp::exist_file(cSetTiePMul::StdName(mICNM,mSH,mPost,true)))
		aPMulFile = cSetTiePMul::StdName(mICNM,mSH,mPost,true);
	else if (ELISE_fp::exist_file(cSetTiePMul::StdName(mICNM,mSH,mPost,false)))
		aPMulFile = cSetTiePMul::StdName(mICNM,mSH,mPost,false);
	else
	{
		std::cout << "cAppliImportSfmInit::SaveCoords() No tie points in " 
				  << cSetTiePMul::StdName(mICNM,mSH,mPost,false) << "\n";

		return false;
	}


	aSetPM->AddFile(aPMulFile);
	std::vector<cSetPMul1ConfigTPM *> aVSetPM = aSetPM->VPMul();

	std::cout << "aVSetPM.size() " << aVSetPM.size() << "\n";

	for (int aK=0; aK<int(aVSetPM.size()); aK++)
	{
		
	}

/*
    {
        cSetPMul1ConfigTPM * aPMConfig = aVSetPM[aK];
        aGes->FillPMulConfigToHomolPack(aPMConfig, mIs2Way);
    }
 * */

	return true;
}

/* Remark: calibrations are considered without distortions! */
bool cAppliImportSfmInit::SaveCC()
{

	//save cc.txt
    std::fstream aCC;
    aCC.open(mCCFile.c_str(), std::istream::out);

	//save list.txt
	std::fstream aL;
	aL.open(mCCListAllFile.c_str(), std::istream::out);

	//read
	int aId=0;
	for (auto aNF : mLFile)
	{
		CamStenope * aCam = mICNM->StdCamStenOfNames(aNF,mMMOriDir);
		std::cout << aCam->PP() << " " << aCam->Focale() << "\n";

		mCC[aId] = new CamSfmInit (aNF,aCam->PP(),aCam->Focale());

		aCC << aId << "\n";
		aL << aNF << " 0 " << aCam->Focale() << "\n";

		aId++;
	}	

	aCC.close();
	aL.close();

	return true;
}

bool cAppliImportSfmInit::Save()
{

	mLFile = mICNM->StdGetListOfFile(mPat,1);

	if (SaveCC())
    	std::cout << "[ImportSfmInit] Save " << mCCFile << " done. " << "\n";

	
	//coords + tracks
	if (SaveCoords())
    	std::cout << "[ImportSfmInit] Save " << mCoordsFile << " and " << mTracksFile << " done. " << "\n";
	//EGs


/*
 *
 *      std::map<int,CamSfmInit *>       mCC;//int follows the SfmInit indexing
        std::map<int,int>                mSfm2MM_ID;//mapping of the sfm indexes to MM indexes; the latter must follow the image order in mCCVec
        std::vector<std::string> *       mCCVec;//Vec to initialise the merge structure
        std::map<int,tKeyPt >            mC2KPts; //map containing a keypoint set per image; the int is the image id as in mCC


 *
 *
 * */


	return true;
}

cAppliImportSfmInit::cAppliImportSfmInit(int argc,char ** argv) :
    DoCalib(false),
    DoImags(false),
    mEGFile(""),
	mRotSolFile(""),
	mTrSolFile(""),
	mMMOriDir(""),
    mSH(""),
    mPost(""),
    mPat(""),
    CalibDir(""),
    mCCVec(new std::vector<std::string>())
{

    ElInitArgMain
    (
        argc,argv,
        LArgMain() << EAMC(mDir,"Working dir. If inside put ./")
                   << EAMC(mCCFile,"cc.txt (SfmInit format)", eSAM_IsExistFile)
                   << EAMC(mCCListAllFile,"list.txt",eSAM_IsExistFile)
                   << EAMC(mCoordsFile,"coords.txt",eSAM_IsExistFile)
                   << EAMC(mTracksFile,"tracks.txt",eSAM_IsExistFile),
        LArgMain() << EAM(mRotSolFile,"Rot","true","rot_solution.txt") 
				   << EAM(mTrSolFile,"Tr","true","trans_problem_solution.txt")
				   << EAM(mEGFile,"EG","true", "Export relative orientations from EGs.txt")
                   << EAM(CalibDir,"OriCalib","true", "Export calibrations to OriCalib directory")
                   << EAM(DoCalib,"DoCal",true,"Export the calibration files; Def=true")
                   << EAM(DoImags,"DoImg",true,"Create images' xml list from cc.txt")
                   << EAM(mSH,"SH",true,"Homol postfix")
				   << EAM(mPost,"Post",true,"PMul${Dest}.txt/dat")
                   << EAM(mMMOriDir,"Ori",true,"If Pat is init -> Export MM to SfmInit else only ori to txt format ")
                   << EAM(mPat,"Pat",true,"Pattern of images to export to SfmInit problem")
    );    

    #if (ELISE_windows)
        replace( mDir.begin(), mDir.end(), '\\', '/' );
    #endif

    mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
  



    /* Read SfmInit problem in MicMac */	
    if (mPat=="")
	{	
        if (EAMIsInit(&CalibDir))
        {
            //StdCorrecNameOrient(CalibDir, mDir);
            
            if (! ELISE_fp::IsDirectory("Ori-"+CalibDir))
                ELISE_fp::MkDir("Ori-"+CalibDir);
 
        }
 	   if (EAMIsInit(&mRotSolFile))
 	   {
 	   	if (! ELISE_fp::IsDirectory("Ori-"+SolutionOriName()))
                ELISE_fp::MkDir("Ori-"+SolutionOriName());
 	   }
 
 	   if (EAMIsInit(&mMMOriDir))
 	   	mMMOriDir= mICNM->StdKeyOrient(mMMOriDir);
	

	   Read();

	}
	/* Export MicMac to SfmInit problem */
	else
	{
		if (EAMIsInit(&mMMOriDir))
			Save();	
		else
			ELISE_ASSERT(false,"You must initialise Ori when Pat is used.")
	}


}

int CPP_NewOriReadFromSfmInit(int argc,char ** argv)
{

    cAppliImportSfmInit aAppli(argc,argv);
    

    return EXIT_SUCCESS;
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
