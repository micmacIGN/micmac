#include "ConvertTiePPs2MM.h"



int ConvertTiePPs2MM_main(int argc,char ** argv)
{

    string mDir="./";
    Pt2dr aC;
    Pt2dr aSize;
    string mPSFile;
    string aOut="_PS";
    string aPat;
    bool a2W = false;

    ElInitArgMain
    (
          argc, argv,
          LArgMain() << EAMC(mDir, "Directory")
                     << EAMC(aC, "[Cx Cy] values from .xml PhotoScan file")
                     << EAMC(aSize, "Size of photosite in mm unit")
                     << EAMC(mPSFile, "PhotoScan ORIMA input File", eSAM_IsExistFile),
          LArgMain() << EAM(aOut,"Out",false,"Output Homol Name ; Def=Homol_PS")
                     << EAM(aPat,"Pat",false,"Pattern of images")
                     << EAM(a2W,"Way",false,"homologue in 2 way - pas encore implementer pour l'instatn", eSAM_IsBool)
     );


 cAppliConvertTiePPs2MM * aAppli = new cAppliConvertTiePPs2MM();
 bool importOK = aAppli->readPSTxtFile( (mDir + mPSFile).c_str(), aAppli->VHomolPS());
 if (importOK)
 {
     aAppli->initAllPackHomol(aAppli->ImgUnique());
     cout<<endl<<endl<<"Begin fusion Homol : "<<endl;
     for (uint aKId=0; aKId<uint(aAppli->IdMaxGlobal()); aKId++)
     {
         if (aAppli->IdMaxGlobal() > 300)
         {
             if (aKId %  (aAppli->IdMaxGlobal()/300) == 0)
                  cout<<"   ++ ["<<(aKId*100.0/aAppli->IdMaxGlobal())<<" %] - fusion"<<endl;
         }
         vector<cHomolPS*> aVItemHaveSameId;
         aAppli->getId(aKId, aVItemHaveSameId);
         aAppli->addToHomol(aVItemHaveSameId, aC, aSize);
     }
     cout<<endl<<endl<<"Begin write Homol : "<<endl;
     aAppli->writeToDisk(aOut,a2W, mDir);
 }
 cout<<endl<<endl<<"********  Finish  **********"<<endl;
 return EXIT_SUCCESS;
}
