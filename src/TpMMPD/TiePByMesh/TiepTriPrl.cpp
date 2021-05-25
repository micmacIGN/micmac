#include "../src/uti_phgrm/TiepTri/TiepTri.h"

int TiepTriPrl_main(int argc,char ** argv)
{
    cout<<"*********************************"<<endl;
    cout<<"* Interface paralell of TiepTri *"<<endl;
    cout<<"*********************************"<<endl;

   std::string aFullNameXML,anOri;
   std::string KeyMasqIm = "NONE";
   int nInt = 0;
   bool NoTif = false;
   int FFS = ((1<<16)-1);
   bool mFilFAST = true;
   bool mFilAC = true;
   double mTT_SEUIL_SURF_TRI = 100;
   double mTT_SEUIL_CORREL_1PIXSUR2 = 0.7;

   double mDistFiltr       = TT_DefSeuilDensiteResul ;
   int mNumInterpolDense  = -1;
   bool mDoRaffImInit       = false;
   int mNbByPix            = 1;
   int mNivLSQM           = -1;
   double mRandomize          = 0.0;
   bool UseABCorrel       = false;

   string mHomolOut = "_TiepTri";
   double mTT_SEUIl_DIST_Extrema_Entier = 1.5;

   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(aFullNameXML, "Pattern of XML for Triangu",  eSAM_IsPatFile)
                     << EAMC(anOri,        "Orientation dir"),
         LArgMain()
                     << EAM(KeyMasqIm, "KeyMasqIm", true, "Key Masq, def=NONE")
                     << EAM(NoTif, "NoTif", true, "No Img TIF, def=false")
                     << EAM(FFS,  "FFS",true,"Flag spatial filtering (tuning) [0 = all OFF, 1 = In Each Tri, 8 = Final in All Tri, 9 = all ON (def) ")
                     << EAM(mFilFAST,  "FilFAST",true,"Use FAST condition ? (def = true)")
                     << EAM(mFilAC,  "FilAC",true,"Use Autocorrelation condition ? (def = true)")
                     << EAM(mTT_SEUIL_SURF_TRI,  "surfTri",true,"Surface min to eliminate too small triangle (def = 100 unit)")
                     << EAM(mTT_SEUIL_CORREL_1PIXSUR2,  "correlBrut",true,"Threshold of correlation score 1pxl/2 (def = 0.7)")
                     << EAM(nInt, "nInt", true, "display command")
               << EAM(mDistFiltr,"DistF",true,"Average distance between tie points")
               << EAM(mNumInterpolDense,"IntDM",true," Interpol for Dense Match, -1=NONE, 0=BiL, 1=BiC, 2=SinC")
               << EAM(mDoRaffImInit,"DRInit",true," Do refinement on initial images, instead of resampled")
               << EAM(mNivLSQM,"LSQC",true,"Test LSQ,-1 None (Def), Flag 1=>Affine Geom, Flag 2=>Affin Radiom")
               << EAM(mNbByPix,"NbByPix",true," Number of point inside one pixel")
               << EAM(mRandomize,  "Randomize",true,"Level of random perturbation (LSQ Match), def=1.0 in interactive, else 0.0  ")
               << EAM(UseABCorrel,  "UseABCorrel",true,"Tuning use correl in mode A*v1+B=v2 ")

               << EAM(mHomolOut,  "Out",true,"Suffix for Homol Out Folder (def=_TiepTri)")
               << EAM(mTT_SEUIl_DIST_Extrema_Entier,  "distWithExtr",true,"Distant limit b/w max correl point & extrema (def=1.5pxl)")

   );

   cout<<"mFilSpatial "<<endl;
   cout<<"mFilSpatial & 8 "<<(FFS & 8)<<endl;
   cout<<"mFilSpatial & 1 "<<(FFS & 1)<<endl;

   std::string aDir,aNameXML;
   SplitDirAndFile(aDir,aNameXML,aFullNameXML);
   string aDirXML = aDir;
   if (!StdCorrecNameOrient(anOri,aDir,true))
   {
      StdCorrecNameOrient(anOri,"./");
      aDir = "./";
   }
   cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aDirXML);
   vector<string> aVXML =  *(anICNM->Get(aNameXML));
   cout<<aVXML.size()<<" xml file"<<endl;
   string aNoTif = NoTif ? "true":"false";
   //paralelliser task
   std::list<std::string> aLCom;
   for (int aK=0 ; aK<int(aVXML.size()) ; aK++)
   {
        std::string aNameFile =  aVXML[aK];
        if (ELISE_fp::exist_file(aDirXML+aNameFile))
        {
            std::ostringstream strs;
            strs << mTT_SEUIL_SURF_TRI;
            std::string str_TT_SEUIL_SURF_TRI = strs.str();
            strs.str("");
            strs << mTT_SEUIL_CORREL_1PIXSUR2;
            std::string str_TT_SEUIL_CORREL_1PIXSUR2 = strs.str();
            strs.str("");
            strs << FFS;
            std::string str_FFS = strs.str();


            std::string aCom = MM3DStr + " TestLib TiepTri " + aDirXML + aNameFile + " " + anOri +
                                " KeyMasqIm=" + KeyMasqIm + " NoTif=" + aNoTif +
                                " FFS=" + str_FFS +
                                " FilFAST=" + (mFilFAST ? "true" : "false") +
                                " FilAC=" + (mFilAC ? "true" : "false") +
                                " surfTri=" + str_TT_SEUIL_SURF_TRI +
                                " correlBrut=" + str_TT_SEUIL_CORREL_1PIXSUR2;
            if (EAMIsInit(&mDistFiltr))
                aCom = aCom + " DistF=" + ToString(mDistFiltr);
            if (EAMIsInit(&mNumInterpolDense))
                aCom = aCom + " IntDM=" + ToString(mNumInterpolDense);
            if (EAMIsInit(&mDoRaffImInit))
                aCom = aCom + " DRInit="+ (mDoRaffImInit ? "true" : "false");
            if (EAMIsInit(&mNivLSQM))
                aCom = aCom + " LSQC=" + ToString(mNivLSQM);
            if (EAMIsInit(&mNbByPix))
                aCom = aCom + " NbByPix=" + ToString(mNbByPix);
            if (EAMIsInit(&mRandomize))
                aCom = aCom + " Randomize=" + ToString(mRandomize);
            if (EAMIsInit(&UseABCorrel))
                aCom = aCom + " UseABCorrel=" + (UseABCorrel ? "true" : "false");
            if (EAMIsInit(&mHomolOut))
                aCom = aCom + " Out=" + mHomolOut;
            if (EAMIsInit(&mTT_SEUIl_DIST_Extrema_Entier))
                aCom = aCom + " distWithExtr=" + ToString(mTT_SEUIl_DIST_Extrema_Entier);


            aLCom.push_back(aCom);
            if (nInt != 0)
                std::cout << aCom << "\n\n";
         }
   }
   cEl_GPAO::DoComInParal(aLCom);
   return EXIT_SUCCESS;

}


