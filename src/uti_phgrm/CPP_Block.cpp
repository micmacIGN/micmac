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



int Blinis_main(int argc,char ** argv)
{
    // MemoArg(argc,argv);
    MMD_InitArgcArgv(argc,argv);

    std::string  aDir,aPat,aFullDir;
    std::string AeroIn;
    std::string KeyIm2Bloc;
    std::string aFileOut;


    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aFullDir,"Full name (Dir+Pat)", eSAM_IsPatFile)
                    << EAMC(AeroIn,"Orientation in", eSAM_IsExistDirOri)
                    << EAMC(KeyIm2Bloc,"Key for computing bloc structure")
                    << EAMC(aFileOut,"File for destination"),
        LArgMain()
    );

    if (!MMVisualMode)
    {
    #if (ELISE_windows)
        replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
    #endif
    SplitDirAndFile(aDir,aPat,aFullDir);
    StdCorrecNameOrient(AeroIn,aDir);

    cStructBlockCam aBlockName = StdGetFromPCP(Basic_XML_MM_File("Stereo-Bloc_Naming.xml"),StructBlockCam);
    aBlockName.KeyIm2TimeCam()  = KeyIm2Bloc;
    ELISE_fp::MkDirSvp(aDir+"Tmp-MM-Dir/");
    MakeFileXML(aBlockName,aDir+"Tmp-MM-Dir/Stereo-Bloc_Naming.xml");
    std::string aCom =   MM3dBinFile_quotes( "Apero" )
                       + Basic_XML_MM_File("Stereo-Apero-EstimBlock.xml ")
                       + std::string(" DirectoryChantier=") +aDir +  std::string(" ")
                       + std::string(" +PatternIm=") + QUOTE(aPat) + std::string(" ")
                       + std::string(" +Ori=") + AeroIn
                       + std::string(" +Out=") + aFileOut
                    ;


    std::cout << "Com = " << aCom << "\n";
    int aRes = System(aCom.c_str(),false,true,true);
    return aRes;

    }
    else return EXIT_SUCCESS;
}

/***********************************************************************/

class cAOFB_Im
{
     public :
        cAOFB_Im(const std::string & aName,CamStenope* aCamInit,const std::string& aNameCalib) :
          mName      (aName),
          mCamInit   (aCamInit),
          mNameCalib  (aNameCalib),
          mDone      (mCamInit!=nullptr),
          mR_Cam2W   (mDone ? mCamInit->Orient().inv()  : ElRotation3D::Id)
        {
        }

        std::string   mName;     // Name of image
        CamStenope  * mCamInit;  // Possible initial camera (pose+Cal)
        std::string   mNameCalib; // Name of calibration
        bool          mDone;
        ElRotation3D  mR_Cam2W;  // Orientation Cam -> Word
};

class cAppli_OriFromBlock
{
    public :
        cAppli_OriFromBlock (int argc,char ** argv);
    private :
        // Extract Time Stamp and Grp from Compiled image
        typedef std::pair<std::string,std::string>  t2Str;
        t2Str  TimGrp(cAOFB_Im * aPtrI)
        {
           return mICNM->Assoc2To1(mKeyBl,aPtrI->mName,true);
        }

        std::string mPatIm;  // Liste of all images
        std::string mOriIn;  // input orientation
        std::string mNameCalib;  // input calibration (required for non oriented image)
        std::string mNameBlock;  // name of the block-blini
        std::string mOriOut;     // Output orientation

        cStructBlockCam mBlock;  // structure of the rigid bloc
        std::string     mKeyBl;  // key for name compute of bloc


        cElemAppliSetFile mEASF; // Structure to extract pattern + name manip
        std::string                        mDir; // Directory of data
        cInterfChantierNameManipulateur *  mICNM;  // Name manip
        const std::vector<std::string> *   mVN;    // list of images
        std::vector<cAOFB_Im*>             mVecI;  // list of "compiled"
        std::map<t2Str,cAOFB_Im*>          mMapI;  // map Time+Grp -> compiled image

};

cAppli_OriFromBlock::cAppli_OriFromBlock (int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv);

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mPatIm,"Full name (Dir+Pat)", eSAM_IsPatFile)
                    << EAMC(mOriIn,"Input Orientation folder", eSAM_IsExistDirOri)
                    << EAMC(mNameCalib,"Calibration folder", eSAM_IsExistDirOri)
                    << EAMC(mNameBlock,"File for block"),
        LArgMain()
    );

        //  =========== Naming and name stuff ====
    mEASF.Init(mPatIm); // Compute the pattern
    mICNM = mEASF.mICNM;  // Extract name manip
    mVN = mEASF.SetIm();  // Extract list of input name
    mDir = mEASF.mDir;

    StdCorrecNameOrient(mOriIn,mDir);
    StdCorrecNameOrient(mNameCalib,mDir);


        //  =========== Block stuff ====
    mBlock = StdGetFromPCP(mNameBlock,StructBlockCam); // Init Block
    mKeyBl =  mBlock.KeyIm2TimeCam(); // Store key for facilitie
    
    // Create structur of "compiled" cam
    for (int aKIm=0 ; aKIm<int(mVN->size()) ; aKIm++)
    {
        const std::string & aName = (*mVN)[aKIm];
        CamStenope * aCamIn = mICNM->StdCamStenOfNamesSVP(aName,mOriIn);  // Camera may not 
        std::string  aNameCal = mICNM->StdNameCalib(mNameCalib,aName);
        mICNM->GlobCalibOfName(aName,mNameCalib,true);  // Calib should exist
        cAOFB_Im * aPtrI = new cAOFB_Im(aName,aCamIn,aNameCal);  // Create compiled
        mVecI.push_back(aPtrI);  // memo all compiled
        mMapI[TimGrp(aPtrI)] = aPtrI; // We will need to access to pose from Time & Group
        // std::pair<std::string,std::string> aPair=mICNM->Assoc2To1(mKeyBl,aPtrI->mName,true);
    }

    // Try to compile pose non existing
    for (const auto & aPtrI : mVecI)
    {
        // If cam init exits nothing to do
        if (! aPtrI->mCamInit)
        {
           int  aNbInit = 0;
           // Extract time stamp & name of this block
           t2Str aPair= TimGrp(aPtrI);
           std::string aNameTime = aPair.first; // get a time stamp
           std::string aNameGrp  = aPair.second;// get a cam name
           
           cParamOrientSHC * OrInBlock = POriFromBloc(mBlock,aNameGrp,false); // Extrac orientaion in block

           // In the case there is several head oriented, we store multiple solutio,
           std::vector<ElRotation3D>  aVecOrient;
           std::vector<double>        aVecPds;

           // Parse the block of camera
           for (const auto & aLiaison :  mBlock.LiaisonsSHC().Val().ParamOrientSHC())
           {
              std::string aNeighGrp = aLiaison.IdGrp();
              if (aNameGrp != aNeighGrp)  // No need to try to init on itself
              {
                 // Extract neighboor in block from time-stamp & group
                 cAOFB_Im* aNeigh = mMapI[t2Str(aNameTime,aNeighGrp)];
                 if (aNeigh && aNeigh->mCamInit) // If neighbooring exist and has orientation init
                 {
                    cParamOrientSHC * aOriNeigh = POriFromBloc(mBlock,aNeighGrp,false);

                    ElRotation3D aOri_Neigh2World =  aNeigh->mCamInit->Orient().inv(); // Monde->Neigh
                    ElRotation3D aOriBlock_This2Neigh = RotCam1ToCam2(*OrInBlock,*aOriNeigh);
                    ElRotation3D aOri_This2Word = aOri_Neigh2World * aOriBlock_This2Neigh;
                    aVecOrient.push_back(aOri_This2Word);
                    aVecPds.push_back(1.0);
                    aNbInit++;
                    std::cout << aPtrI->mName << " From " << aNeigh->mName << "\n";
                 }
              }
           }
           if (aNbInit)
           {
              aPtrI->mR_Cam2W = AverRotation(aVecOrient,aVecPds);
              aPtrI->mDone = true;
           }
           else
           {
           }
        }
    }

    // Now export 
    for (const auto & aPtrI : mVecI)
    {
        if (aPtrI->mDone)
        {
            // aPtrI->mCamCalib->SetOrientation(aPtrI->mR_Cam2W.inv());
 // std::string StdExport2File(cInterfChantierNameManipulateur *,const std::string & aDirOri,const std::string & aNameIm);
        }
    }

    // cElemAppliSetFile anEASF(mPatIm);
}

int OrientFromBlock_main(int argc,char ** argv)
{
    cAppli_OriFromBlock anAppli(argc,argv);

    return EXIT_SUCCESS;
}


/***********************************************************************/

class cOneImBrion
{
    public :
    private :
        std::string  mName; 
        CamStenope   * mCalib;
};

class cAppli_Brion   // Block Rigid Initialisation des Orientation Normale
{
    public :
    private :
       cStructBlockCam  mBlock;
};



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
