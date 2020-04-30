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

#include "TestNewOri.h"

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

//read artsquad benchmark
class cAppliImportArtsQuad;

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

template <typename T>
void FileReadOK(FILE *fptr, const char *format, T *value)
{
    int OK = fscanf(fptr, format, value);
    if (OK != 1)
        ELISE_ASSERT(false, "cAppliImportSfmInit::FileReadOK")

}


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
        void ConvMMConv(Pt2dr &aPt,bool IsDirect);
		void ConvRelMM2SfmI(ElRotation3D& aOri2in1, ElMatrix<double>& aRij, Pt3dr& atij);
		void InitRotZ();


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
		bool SaveCoords();
		void SaveEG();

		ElRotation3D OriCam2On1(const cNewO_NameManager *aNM,const std::string & aNOri1,const std::string & aNOri2,bool & OK) const;

 
        //template <typename T>
        //void FileReadOK(FILE *fptr, const char *format, T *value);

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

		std::map<std::string,int>        mStr2Id;//mapping from image name to id
        //cVirtInterf_NewO_NameManager *   mVNM;
		

		ElMatrix<double>*   mRotZ;
		

};

/*template <typename T>
void cAppliImportSfmInit::FileReadOK(FILE *fptr, const char *format, T *value)
{
    int OK = fscanf(fptr, format, value);
    if (OK != 1)
        ELISE_ASSERT(false, "cAppliImportSfmInit::FileReadOK")

}*/

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

/* MM:      	 origin at (0,0)
   SfmInit/SIFT: origin at (1,1) */
void cAppliImportSfmInit::ConvMMConv(Pt2dr &aPt,bool IsDirect)
{
	/* From SIFT to MM */
	if (IsDirect)
	{
		aPt.x = aPt.x -1;
    	aPt.y = aPt.y -1;
	}
	/* From MM to SIFT */
	else
	{
		aPt.x = aPt.x +1;
    	aPt.y = aPt.y +1;
	
	}
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
            ConvMMConv(aPt,true);
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

/* For now I do not need to read edges 
 * code to finish when of interest */

/*
 *INPUT:
 *         Rimm is the identity affine rotation and describes cam i location and rotation
 *         Rjmm is the location of cam j in camera i
 *
 *OUTPUT:
 *         Rij_sfm
 *         tij_sfm
 *
 *
 *
 *CONVERSION SCRIPT :
 *
 *         Risfm0 = Rimm * RotZ   //where Rimm is Id
 *         Rjsfm0 = Rjmm * RotZ
 *
 *         Risfm1 = Risfm0.transpose()
 *         Rjsfm1 = Rjsfm0.transpose()
 *
 *
 *    Relative rotation in MicMac convention :
 *          Rij_sfm = Risfm1 * Rjsfm1.transpose()
 *
 *    Relative translation from i to j in world coords :
 *          tij_world = (ti - tj) / norm(ti - tj)
 *
 *    Relative translation from i to j in cam i's coords
 *          tij_sfm   = Risfm1 * tij_world   => tij_world = Risfm' * tij_sfm
 *
 *                                            Risfm' * tij_sfm = (ti - tj) / norm(ti - tj)
 *                                               ti = (0,0,0)
 *                                               Rimm = Id
 *
 *                                            Risfm1' * tij_sfm = -tj 
 *                                       	  tj = - (Risfm1.transpose() * tij_sfm) 
 *											  tj = - (Risfm0.transpose().transpose() * tij_sfm )
 *											  tj = - (Risfm0 * tij_sfm )
 *											  tj = - (Rimm * RotZ * tij_sfm)
 *										->	  tj = - (RotZ * tij_sfm)
 *
 *                                            Rij_sfm = Risfm1 * Rjsfm1.transpose()
 *                                            Rij_sfm = (Risfm0.transpose()) * (Rjsfm0.transpose()).transpose()
 *                                            Rij_sfm =  (Rimm * RotZ).transpose() *
 *                                                       ((Rjmm * RotZ).transpose()).transpose()
 *
 *                                            Rij_sfm = RotZ.transpose() * Rimm.transpose() * 
 *                                                      (Rjmm * RotZ)
 *
 *                                            RotZ * Rij_sfm = Rimm.transpose() * Rjmm * RotZ
 *
 *                                            Rimm.transpose() = Id
 *                                            RotZ * Rij_sfm * RotZ.transpose() = Id * Rjmm
 *                                       ->   Rjmm = RotZ * Rij_sfm * RotZ.transpose()
 *
 *
 *
 *==================================================
 *      
 *         */
bool cAppliImportSfmInit::ReadEdges()
{


    FILE* fptr = fopen(mEGFile.c_str(), "r");
    if (fptr == NULL) {
      return false;
    };

	int aNbEdge=0;
    while (!std::feof(fptr)  && !ferror(fptr))
    {
        Pt2di aE;
        Pt3dr            atij_sfm;
        Pt3dr            atj;
        ElMatrix<double> aRij_sfm(3,3,1.0);
        ElMatrix<double> aRjmm(3,3,1.0);

        bool OK = std::fscanf(fptr, "%i %i %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", 
                          &aE.x, &aE.y, 
                          &aRij_sfm(0,0), &aRij_sfm(0,1), &aRij_sfm(0,2),
                          &aRij_sfm(1,0), &aRij_sfm(1,1), &aRij_sfm(1,2),
                          &aRij_sfm(2,0), &aRij_sfm(2,1), &aRij_sfm(2,2),
                          &atij_sfm.x, &atij_sfm.y, &atij_sfm.z);

		
		
		//atj = - ((*mRotZ) * atij_sfm);
		atj = ((*mRotZ) * atij_sfm);
		aRjmm = ((*mRotZ) * aRij_sfm) * (*mRotZ).transpose();


	 
        if (0)
		{


			atij_sfm.x = -0.26035493;
			atij_sfm.y = 0.20397937;
			atij_sfm.z = 0.94372015;

			aRij_sfm(0,0) = 0.99054416;
			aRij_sfm(0,1) = 0.02618604;
			aRij_sfm(0,2) = 0.134672;
			aRij_sfm(1,0) = -0.01386052;
			aRij_sfm(1,1) = 0.99569403;
			aRij_sfm(1,2) = -0.09165854;
			aRij_sfm(2,0) = -0.13649229;
			aRij_sfm(2,1) = 0.08892521;
			aRij_sfm(2,2) = 0.98664187;

			atj = - ((*mRotZ) * atij_sfm);
	        aRjmm = ((*mRotZ) * aRij_sfm) * (*mRotZ).transpose();


			std::cout << aRjmm(0,0) << " " << aRjmm(0,1) << " " << aRjmm(0,2) << "\n"
                      << aRjmm(1,0) << " " << aRjmm(1,1) << " " << aRjmm(1,2) << "\n"
                      << aRjmm(2,0) << " " << aRjmm(2,1) << " " << aRjmm(2,2) << "\n";
        	std::cout << "T " << atj << " E " << aE << " " << OK << "\n";


			getchar();

		}


		cNewO_NameManager *aNM = new cNewO_NameManager("",mSH,true,mDir,CalibDir,"dat");
	    cSauvegardeNamedRel aLCpl;

		std::string aN1 = mCC[aE.x]->mName;
		std::string aN2 = mCC[aE.y]->mName;

		std::string aNameXML    = aNM->NameXmlOri2Im(aN1,aN2,false);
		std::string aNameXMLBin = aNM->NameXmlOri2Im(aN1,aN2,true);
	
		cXml_Ori2Im aXml;
		aXml.Im1() = aN1;
		aXml.Im2() = aN2;

		if (CalibDir == "")
			ELISE_ASSERT(false, "ReadEdges() Choose a CalibDir")
		else
		{
			aXml.Calib() = CalibDir;

			//the relative motion
			cXml_O2IComputed aRMC;
			cXml_O2IRotation aRAff;
			
			cTypeCodageMatr aTCRot;
			aTCRot.L1() = Pt3dr(aRjmm(0,0),aRjmm(0,1),aRjmm(0,2));   
        	aTCRot.L2() = Pt3dr(aRjmm(1,0),aRjmm(1,1),aRjmm(1,2));   
        	aTCRot.L3() = Pt3dr(aRjmm(2,0),aRjmm(2,1),aRjmm(2,2));   
        	aTCRot.TrueRot() = true;

			aRAff.Ori() = aTCRot;
			aRAff.Centre() = atj;

			aRMC.OrientAff() = aRAff;
			
			aXml.Geom() = aRMC;	
		}

		MakeFileXML(aXml,aNameXML);
		MakeFileXML(aXml,aNameXMLBin);
		
		//std::cout << aNbEdge << ".." << aNameXML << " " ;
		aNbEdge++;
    }
	std::cout << "\n";

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
/* 1/ Read keypts per camera and update io
   2/ Read tracks and create Homol

   Decoding coords, eg:
   2 0 0 1 0 
    - a tie-pts visible in 2 images
    - image id 0, point id 0
    - image id 1, point id 0 */

/*    coords list of Pts per camera    => map<id_cam,tkey*>
 *    tracks: id cam, id pts in list   => map<track_id,vector<pairs<id_cam,pt_id>>* >
 * 
 *    for each conf
 *        for each pt
 *            get cam for that pt
 *            add that pt to coord_list for the cam ; retrive tkey and push next tkey_pair <pt_id,pt2dr> id follows an order
 *            add that pt to track : track_id=conf, push pair<id_cam,pt_id> 
 *
 * */
/* Remark: calibrations are considered without distortions! */

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

	int aNbConf = aVSetPM.size(); 


	/* Initalise mC2KPts and aTracks  */
	std::map<int,std::vector<std::pair<int,int> > > aTracks;

	//aK corresponds to configurations and each typically contains more than one track
	int aTrkGlob=0;
	for (int aK=0; aK<aNbConf; aK++)
	{
		cSetPMul1ConfigTPM * aOneConf = aVSetPM[aK];

		std::vector<int> aVImId = aOneConf->VIdIm();
		int aNbCamInConf = int(aVImId.size());
    	int aNbPtInConf = aOneConf->NbPts();

		std::cout << "Track: " << aK << " nb pts: " << aNbPtInConf << " nb im:" << aVImId.size() << "\n";


		for (int aPId=0; aPId<aNbPtInConf; aPId++)
		{
			//vector containing a single track (first is camera id, second is points id)
        	std::vector<std::pair<int,int> > *aVPt = &(aTracks[aTrkGlob]);

			for (int aPIm1=0; aPIm1<aNbCamInConf; aPIm1++)
        	{
				int aImId1 = aVImId[aPIm1];
            	Pt2dr aPt1 = aOneConf->GetPtByImgId(aPId, aImId1);

				//get current keypoints for that image
				tKeyPt *aCurKey  = &(mC2KPts[aImId1]); 

				//serves as the id of the point
				int aKeySz = int(aCurKey->size());


				//save to keypoints of current points
				//	+add new pt at the end of the key structure
				tKeyPt::iterator aKeyEnd = aCurKey->end();
				aCurKey->insert(aKeyEnd, std::pair<int,Pt2dr>(aKeySz,aPt1));

				//save to current track
				std::pair<int,int> aCamPtId(aImId1,aKeySz);
				aVPt->push_back (aCamPtId);


				//std::cout << "Pt= " << aPt1 << " " << aKeySz << " cam: " << aImId1 << "\n";


			}
		    aTrkGlob++;
		
		}

		
	}


    /* Save tracks.txt */
    std::fstream aTrkStream;
    aTrkStream.open(mTracksFile.c_str(), std::istream::out);

	aTrkStream << aTrkGlob << "\n";

	for (auto aT : aTracks)
	{
		//length of the track
		int aNbMul = int(aT.second.size());

		aTrkStream << aNbMul << " " ;

		//the track itself
		for (int aMul=0; aMul<aNbMul; aMul++)
		{
			//          cam_id                              pt_id 
			aTrkStream << aT.second.at(aMul).first << " " << aT.second.at(aMul).second << " " ;
		}
		aTrkStream << "\n"; 

	}

	aTrkStream.close();
	std::cout << "tracks saved " << "\n";

	/* Save coords.txt AND fill the mCC map */
	std::fstream aCoordStream;
    aCoordStream.open(mCoordsFile.c_str(), std::istream::out);

	for (auto aIm : mC2KPts)
	{
		std::string aImName = aSetPM->NameFromId(aIm.first);

		std::string aImOriName = mICNM->Dir() + mICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+mMMOriDir,aImName,true);
		
		if (1)
			std::cout << aImOriName << "\n";

		if (  ELISE_fp::exist_file(aImOriName))
		{	
			CamStenope * aCam = mICNM->StdCamStenOfNames(aImName,mMMOriDir);
			mCC[aIm.first] = new CamSfmInit (aImName,aCam->PP(),aCam->Focale());
            
			//initialise the mStr2Id used in SaveEG
			mStr2Id[aImName] = aIm.first;
            
			aCoordStream << "#index = " << aIm.first << ", name = " << aImName << ", keys = " << aIm.second.size() 
					     << ", px = " << aCam->PP().x << ", py = " << aCam->PP().y << ", focal = " << aCam->Focale() << "\n";
            
			for (int aPt = 0; aPt<int(aIm.second.size()); aPt++)
			{
				ConvMMConv(aIm.second.at(aPt),false);
				aCoordStream << aPt << " " << aIm.second.at(aPt).x << " " << aIm.second.at(aPt).y << " 0 0 255 255 255\n";  
            
			}
		}
		else
			std::cout << "cAppliImportSfmInit::SaveCoords() no orientation for " << aImName << " in " << mMMOriDir << "\n"; 
	}	

	aCoordStream.close();


	/* Save cc.txt and list.txt from mC2KPts */
    std::fstream aCC;
    aCC.open(mCCFile.c_str(), std::istream::out);

    //save list.txt
    std::fstream aL;
    aL.open(mCCListAllFile.c_str(), std::istream::out);



    //read
	for (auto aIm : mC2KPts)
    {
		std::string aImName = aSetPM->NameFromId(aIm.first);
		std::string aImOriName = mICNM->Dir() + mICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+mMMOriDir,aImName,true);
		if (  ELISE_fp::exist_file(aImOriName))
		{
			aCC << aIm.first << "\n";
        	aL <<  mCC[aIm.first]->mName << " 0 " << mCC[aIm.first]->F << "\n";
		}
		else
        	aL <<  aImName << "\n";
			


    }

    aCC.close();
    aL.close();	


		

	return true;
}

ElRotation3D cAppliImportSfmInit::OriCam2On1(const cNewO_NameManager *aNM,const std::string & aNOri1,const std::string & aNOri2,bool & OK) const
{
    OK = false;

    std::string aN1 =  aNOri1;
    std::string aN2 =  aNOri2;
	
	//std::cout << "eeeeeeeeeeeeee " << aNM->NameXmlOri2Im(aN1,aN2,true) << "\n";
	//getchar();

    if (!  ELISE_fp::exist_file(aNM->NameXmlOri2Im(aN1,aN2,true)))
       return ElRotation3D::Id;


    cXml_Ori2Im  aXmlO = aNM->GetOri2Im(aN1,aN2);
    OK = aXmlO.Geom().IsInit();
    if (!OK)
       return ElRotation3D::Id;
    const cXml_O2IRotation & aXO = aXmlO.Geom().Val().OrientAff();
    ElRotation3D aR12 =    ElRotation3D (aXO.Centre(),ImportMat(aXO.Ori()),true);

    OK = true;
    return aR12;

}


/*        
 *INPUT:
 *         Rimm is the identity affine rotation and describes cam i location and rotation
 *         Rjmm is the location of cam j in camera i
 *       
 *         ti is the origin of the coord system and is at (0,0,0)
 *         tj is the location of j in i
 *
 *OUTPUT:
 *         Rij_sfm 
 *         tij_sfm
 *         
 *        
 *        
 *CONVERSION SCRIPT :
 *
 *         Risfm = Rimm * RotZ   //where Rimm is Id
 *         Rjsfm = Rjmm * RotZ
 *        
 *         Risfm = Risfm.transpose()
 *         Rjsfm = Rjsfm.transpose()
 *        
 *        
 *    Relative rotation in MicMac convention :
 *          Rij_sfm = Risfm * Rjsfm.transpose()
 *        
 *    Relative translation from i to j in world coords :
 *          tij_world = (ti - tj) / norm(ti - tj)
 *        
 *    Relative translation from i to j in cam i's coords
 *          tij_sfm   = Risfm * tij_world
 *        
 *         
 *         */

void cAppliImportSfmInit::ConvRelMM2SfmI(ElRotation3D& aOri2in1, ElMatrix<double>& aRij, Pt3dr& atij)
{

	Pt3dr ati(0,0,0);	
	ElMatrix<double> aRi(3,3,0);
	aRi(0,0) = 1;
    aRi(1,1) = 1;
    aRi(2,2) = 1;

	Pt3dr atj(aOri2in1.tr());
	ElMatrix<double> aRj(3,3,0);

	//rotation of Z and inverse
	aRi = aRi * (*mRotZ);
	aRj = aOri2in1.Mat() * (*mRotZ);
	aRi = aRi.transpose();
	aRj = aRj.transpose();

	//relative rotation
	aRij = aRi * aRj.transpose();

	//Pt3dr tij_world = (ati - atj) / euclid(ati - atj);
	Pt3dr tij_world = -(ati - atj) / euclid(ati - atj);
	
	//relative translation
	atij = aRi * tij_world;


}

void cAppliImportSfmInit::SaveEG()
{

	cNewO_NameManager *aNM = new cNewO_NameManager("",mSH,true,mDir,CalibDir,"dat");
	cSauvegardeNamedRel aLCpl;


	std::string aNameLCple = aNM->NameListeCpleOriented(true);

    aLCpl = StdGetFromSI(aNameLCple,SauvegardeNamedRel);

    /* Save to EG.txt */
	std::fstream aEG;
    aEG.open(mEGFile.c_str(), std::istream::out);


	for (auto a2 : aLCpl.Cple())
    {
		

		//Relative ori in MM format
        bool OK;
        ElRotation3D       aR2in1(OriCam2On1 (aNM,a2.N1(),a2.N2(),OK));

		//relative ori in SFMInit format		
		ElMatrix<double>   aRij(3,3,1);
		Pt3dr              atij;

		ConvRelMM2SfmI(aR2in1,aRij,atij);


		//save the relative motion
		if (DicBoolFind(mStr2Id,a2.N1()) && DicBoolFind(mStr2Id,a2.N2()))
		{	
			aEG << mStr2Id[a2.N1()] << " " << mStr2Id[a2.N2()]  << " " 
			    << aRij(0,0) << " " << aRij(1,0) << " " << aRij(2,0) << " "
				<< aRij(0,1) << " " << aRij(1,1) << " " << aRij(2,1) << " "
				<< aRij(0,2) << " " << aRij(1,2) << " " << aRij(2,2) << " "
				<< atij.x     << " " << atij.y     << " " << atij.z     << "\n";
		}
		else
			std::cout << "Could not find ids for " << a2.N1() << " " << a2.N2() << " probably missing in your Ori\n";
    }

	aEG.close();


    std::cout << "Pairs no: " << aLCpl.Cple().size() << "\n";
    std::cout << "aNameLCple: " << aNameLCple << "\n";





}


bool cAppliImportSfmInit::Save()
{

	mLFile = mICNM->StdGetListOfFile(mPat,1);


	
	/* Save coords, tracks, cc and list */
	if (SaveCoords())
    	std::cout << "[ImportSfmInit] Save " << mCoordsFile << " " << mTracksFile << " " << mCCFile << " " << mCCListAllFile << " done. " << "\n";



	/* Save EGs */ 
	if (mEGFile!="")
	{
		SaveEG();	
    	std::cout << "[ImportSfmInit] Save " << mEGFile << " done. " << "\n";
	}


	return true;
}

void cAppliImportSfmInit::InitRotZ()
{
	mRotZ = new ElMatrix<double>(3,3,0);
    (*mRotZ)(0,0) = 1;
    (*mRotZ)(1,1) =-1;
    (*mRotZ)(2,2) =-1;

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
                   << EAM(DoCalib,"DoCal",true,"Export the calibration files; Def=false")
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
  

	//Initialize the RotZ
	InitRotZ();

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

//================================================= ArtsQuad

bool cAppliImportArtsQuad::SaveCalib()
{

	std::string CalibName = "Calib";
	ELISE_fp::MkDirSvp("Ori-"+CalibName);

	std::cout << mFocalName << "\n";
	FILE* fptr = fopen(mFocalName.c_str(), "r");
    if (fptr == NULL) {
          return false;
    };

	std::cout << "START" << "\n";
    char buf[50]; 
	double f;
    while (fscanf(fptr,"%s %lf",buf,&f)==2) 
    {

		Tiff_Im aTF = Tiff_Im::StdConvGen(buf,-1,true);
	   	Pt2di aSzIm = aTF.sz();
        
		cCalibrationInternConique aCIO = StdGetObjFromFile<cCalibrationInternConique>
                (
                    Basic_XML_MM_File("Template-Calib-Basic.xml"),
                    StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                    "CalibrationInternConique",
                    "CalibrationInternConique"
                );

		aCIO.PP()   = Pt2dr(double(aSzIm.x)*0.5,double(aSzIm.y)*0.5) ;
        aCIO.F()    = f ;
        aCIO.SzIm() = aSzIm;
        aCIO.CalibDistortion()[0].ModRad().Val().CDist() = Pt2dr(0,0);

        MakeFileXML(aCIO,mICNM->StdNameCalib(CalibName,buf));

        
    }
    fclose(fptr);



	return true;

}


bool cAppliImportArtsQuad::ImportTracks()
{
	// Read the list of images
	std::vector<std::string> * aVIm = new std::vector<std::string>();

	FILE* fptr = fopen(mListName.c_str(), "r");
    if (fptr == NULL) {
          return false;
    };

	char buf[100]; 
    while (fscanf(fptr,"%s",buf)==1) 
	{
		aVIm->push_back(buf);
        //printf("%s\n", buf); 
	}
	fclose(fptr);

	// Read & save the tracks
	cSetTiePMul * aPMul = new cSetTiePMul(0, aVIm);
	vector<float> vAttr;

	fptr = fopen(mTrackName.c_str(), "r");
    if (fptr == NULL) {
          return false;
    };

	int num_tracks;
    FileReadOK(fptr, "%i", &num_tracks);
    std::cout << "&NbTrk " << num_tracks << "\n";

	/* Iterate over tracks */
    for (int aT=0; aT<num_tracks; aT++)
    {

		std::vector<int>   VImId;
		std::vector<Pt2dr> VImPos;


		int num_views;
        FileReadOK(fptr, "%i", &num_views);

		/* Colect the track aT */
        for (int aK=0; aK<num_views; aK++)
        {
            int 	view_id;
			int 	key_id;
			Pt2dr   xy;
			Pt3di   rgb;


			FileReadOK(fptr, "%i", &view_id);
			FileReadOK(fptr, "%i", &key_id);
			FileReadOK(fptr, "%lf", &xy.x);
			FileReadOK(fptr, "%lf", &xy.y);
			FileReadOK(fptr, "%i", &rgb.x);
			FileReadOK(fptr, "%i", &rgb.y);
			FileReadOK(fptr, "%i", &rgb.z);

			/*std::cout << "view_id " << view_id 
					  << ", xy " << xy << ", rgb " << rgb << "\n";*/

			VImId.push_back(view_id);
			VImPos.push_back(xy);

		}
		aPMul->AddPts(VImId,VImPos,vAttr);
	}

	fclose(fptr);

	std::string aSave = cSetTiePMul::StdName(mICNM,mSH,"",false);
	aPMul->Save(aSave);
	std::cout << "Saved to " << aSave << "\n";

	return true;
}

cAppliImportArtsQuad::cAppliImportArtsQuad(int argc,char ** argv) :
	mFocalName(""),
	mSH("")
{
	ElInitArgMain
    (
        argc,argv,
        LArgMain() << EAMC(mTrackName,"File containing the tracks", eSAM_IsExistFile)
        	       << EAMC(mListName,"File containing the list of images", eSAM_IsExistFile),
        LArgMain() << EAM(mSH,"SH",true,"Homol folder postfix, Def=""")
				   << EAM(mFocalName,"Foc",true,"File containing focal lengths, Def=""")
    );

	mICNM = cInterfChantierNameManipulateur::BasicAlloc("./");

}

int CPP_ImportArtsQuad(int argc,char ** argv)
{
	cAppliImportArtsQuad aAppliAQ(argc,argv);

	//aAppliAQ.ImportTracks();
	aAppliAQ.SaveCalib();

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
