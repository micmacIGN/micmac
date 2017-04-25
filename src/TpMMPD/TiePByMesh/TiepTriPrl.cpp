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
   bool mFilSpatial = true;
   bool mFilFAST = true;
   bool mFilAC = true;
   double mTT_SEUIL_SURF_TRI = 100;
   double mTT_SEUIL_CORREL_1PIXSUR2 = 0.7;

   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(aFullNameXML, "Pattern of XML for Triangu",  eSAM_IsPatFile)
                     << EAMC(anOri,        "Orientation dir"),
         LArgMain()
                     << EAM(KeyMasqIm, "KeyMasqIm", true, "Key Masq, def=NONE")
                     << EAM(NoTif, "NoTif", true, "No Img TIF, def=false")
                     << EAM(mFilSpatial,  "FilSpatial",true,"Use filter spatial ? (def = true)")
                     << EAM(mFilFAST,  "FilFAST",true,"Use FAST condition ? (def = true)")
                     << EAM(mFilAC,  "FilAC",true,"Use Autocorrelation condition ? (def = true)")
                     << EAM(mTT_SEUIL_SURF_TRI,  "surfTri",true,"Surface min to eliminate too small triangle (def = 100 unit)")
                     << EAM(mTT_SEUIL_CORREL_1PIXSUR2,  "correlBrut",true,"Threshold of correlation score 1pxl/2 (def = 0.7)")
                     << EAM(nInt, "nInt", true, "display command")

   );

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


            std::string aCom = MM3DStr + " TestLib TiepTri " + aDirXML + aNameFile + " " + anOri +
                                " KeyMasqIm=" + KeyMasqIm + " NoTif=" + aNoTif +
                                " FilSpatial=" + (mFilSpatial ? "true" : "false") +
                                " FilFAST=" + (mFilFAST ? "true" : "false") +
                                " FilAC=" + (mFilAC ? "true" : "false") +
                                " surfTri=" + str_TT_SEUIL_SURF_TRI +
                                " correlBrut=" + str_TT_SEUIL_CORREL_1PIXSUR2;
            aLCom.push_back(aCom);
            if (nInt != 0)
                std::cout << aCom << "\n\n";
         }
   }
   cEl_GPAO::DoComInParal(aLCom);
   return EXIT_SUCCESS;

}


