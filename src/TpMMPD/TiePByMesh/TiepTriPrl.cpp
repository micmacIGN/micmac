#include "../src/uti_phgrm/TiepTri/TiepTri.h"

int TiepTriPrl_main(int argc,char ** argv)
{
    cout<<"*********************************"<<endl;
    cout<<"* Interface paralell of TiepTri *"<<endl;
    cout<<"*********************************"<<endl;

   std::string aFullNameXML,anOri;
   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(aFullNameXML, "Pattern of XML for Triangu",  eSAM_IsPatFile)
                     << EAMC(anOri,        "Orientation dir"),
         LArgMain()

   );

   std::string aDir,aNameXML;
   SplitDirAndFile(aDir,aNameXML,aFullNameXML);
   string aDirXML = aDir;
   if (!StdCorrecNameOrient(anOri,aDir,true))
   {
      StdCorrecNameOrient(anOri,"./");
      aDir = "./";
   }
   cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
   vector<string> aVXML =  *(anICNM->Get(aNameXML));
   cout<<aVXML.size()<<" xml file"<<endl;

   //paralelliser task
   std::list<std::string> aLCom;
   for (int aK=0 ; aK<int(aVXML.size()) ; aK++)
   {
        std::string aNameFile =  aVXML[aK];
        if (ELISE_fp::exist_file(aNameFile))
        {
            std::string aCom = MM3DStr + " TestLib TiepTri " + aDirXML + aNameFile + " " + anOri;
            aLCom.push_back(aCom);
            //std::cout << aCom << "\n\n";
         }
   }
   cEl_GPAO::DoComInParal(aLCom);
   return EXIT_SUCCESS;

}


