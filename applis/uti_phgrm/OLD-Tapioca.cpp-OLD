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

    MicMa cis an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/
#include "StdAfx.h"


// bin/Tapioca MulScale "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" 300 -1 ExpTxt=1
// bin/Tapioca All  "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" -1  ExpTxt=1
// bin/Tapioca Line  "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" -1   3 ExpTxt=1
// bin/Tapioca File  "../micmac_data/ExempleDoc/Boudha/MesCouples.xml" -1  ExpTxt=1

#define DEF_OFSET -12349876

int  ExpTxt=0;
int  ByP=-1;
std::string aDir,aPat,aPatOri;
std::string aPat2="";
std::string aFullDir; 
int aFullRes;
cInterfChantierNameManipulateur * anICNM =0;
std::string BinPastis;
std::string MkFT;
std::string PostFix;
std::string TheType = "XXXX";

std::string StrMkT() { return (ByP ? (" MkF=" + MkFT +" ") : "") ; }

#define aNbType 4
std::string  Type[aNbType] = {"MulScale","All","Line","File"};

/*
void StdAdapt2Crochet(std::string & aStr)
{
    if (aStr=="") return;

    if ((aStr.find('@')!=(std::string::npos)) && (TheType != Type[3]))
    {
         aStr = "[["+aPat+"]]";
    }
}
*/
void StdAdapt2Crochet(std::string & aStr)
{

    if (TheType != Type[3])
    {
         GlobStdAdapt2Crochet(aStr);
    }
}


std::string RelAllIm()
{

   if (aPat2=="")
     return QUOTE(std::string("NKS-Rel-AllCpleOfPattern@")+ aPat) + std::string(" ");

   return  QUOTE(std::string("NKS-Rel-AllCpleOf2Pat@")+ aPat +"@"+aPat2) + std::string(" ");
}



std::string NKS() 
{ 
    return 
             std::string(" NKS=NKS-Assoc-CplIm2Hom@") 
           + std::string(PostFix)
           + std::string("@")
           + std::string(ExpTxt? "txt" :  "dat") ;
}

void DoMkT()
{
    if (ByP)
    {
		std::string aSMkSr = std::string("make all -f ") + MkFT +std::string(" -j")+ToString(ByP);
        System(aSMkSr,true);
    }
}






void DoDevelopp(int aSz1,int aSz2)
{
    std::list<std::string> aList = anICNM->StdGetListOfFile(aPatOri,1);
	
    cEl_GPAO  aGPAO;
	
    for (std::list<std::string>::iterator iT= aList.begin() ; iT!=aList.end() ; iT++)
    {
        std::string  aNOri = anICNM->Dir()+*iT;
        std::string  aNTif = NameFileStd(aNOri,1,false,true,false);

        std::string aCom = MMBin() + "PastDevlop " + aNOri + " Sz1=" +ToString(aSz1) + " Sz2="+ToString(aSz2);
		
        aGPAO.GetOrCreate(aNTif,aCom);
        aGPAO.TaskOfName("all").AddDep(aNTif);
    }
	
    aGPAO.GenerateMakeFile(MkFT);
	
    DoMkT();
}



int MultiECh(int argc,char ** argv)
{
    int aSsRes;
    int aNbMinPt=2;
    int DoLowRes = 1;

    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAMC(aFullDir,"Full Name (Dir+Pat)")
                     <<EAM(aSsRes,"Size of Low Resolution image")
                     <<EAM(aFullRes,"Siez of High Resolution Images"),
	LArgMain()  << EAM(ExpTxt,"ExpTxt",true)	
                    << EAM(ByP,"ByP",true)	
                    << EAM(PostFix,"PostFix",true)	
                    << EAM(aNbMinPt,"NbMinPt",true)	
                    << EAM(DoLowRes,"DLR",true,"Do Low Resolution")	
                    << EAM(aPat2,"Pat2",true)	
    );

    if (aFullRes != -1)
    {
        // std::cout << "===== EURO COW DO YOU HAVE THIS MESSAGE ?? ==\n";
        std::cout << "Ss-RES = " << aSsRes << " ; Full-Res=" << aFullRes << "\n";
        ELISE_ASSERT(aFullRes>aSsRes,"High Res < Low Res, Probably 2 swap !!");
    }

    StdAdapt2Crochet(aPat2);

    DoDevelopp(aSsRes,aFullRes);


    if (DoLowRes)
    {
         std::string aSsR = 
                        BinPastis
                     +  aDir + std::string(" ") 
                     +  RelAllIm()     //   +  QUOTE(std::string("NKS-Rel-AllCpleOfPattern@")+ aPat) + std::string(" ")
                     +  ToString(aSsRes) + std::string(" ")
                     +  std::string(" NKS=NKS-Assoc-CplIm2Hom@_SRes@dat")
                     +  StrMkT()
                     +  std::string("NbMinPtsExp=2 ")
                     +  std::string("SsRes=1 ");

         System(aSsR,true);
         DoMkT();
    }


    std::string aSFR =  BinPastis
                     +  aDir + std::string(" ") 
                     + QUOTE(std::string("NKS-Rel-SsECh@")+ aPat+ std::string("@")+ToString(aNbMinPt)) + std::string(" ")
                     +  ToString(aFullRes) + std::string(" ")
                     +  StrMkT()
                     +  std::string("NbMinPtsExp=2 ")
                     +  NKS();

    System(aSFR,true);
    DoMkT();

    return 0;
}

int All(int argc,char ** argv)
{
    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAMC(aFullDir,"Full Name (Dir+Pat)")
                     <<EAMC(aFullRes,"Size of image"),
	LArgMain()  << EAM(ExpTxt,"ExpTxt",true)	
                    << EAM(PostFix,"PostFix",true)	
                    << EAM(ByP,"ByP",true)	
                    << EAM(aPat2,"Pat2",true)	
    );
	
    StdAdapt2Crochet(aPat2);
    DoDevelopp(-1,aFullRes);
	
    std::string aSFR =  BinPastis
                     +  aDir + std::string(" ")
                     +  RelAllIm()     //   +  QUOTE(std::string("NKS-Rel-AllCpleOfPattern@")+ aPat) + std::string(" ")
                     +  ToString(aFullRes) + std::string(" ")
                     +  StrMkT()
                     +  std::string("NbMinPtsExp=2 ")
                     +  NKS();
	
    System(aSFR,true);
	
    DoMkT();
	
    return 0;
}

int Line(int argc,char ** argv)
{
    int  aNbAdj;
    int isCirc=0;

    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAMC(aFullDir,"Full Name (Dir+Pat)")
                     <<EAMC(aFullRes,"Size of image")
                     <<EAM(aNbAdj,"Number of ajdcent images to look for"),
	LArgMain()  << EAM(ExpTxt,"ExpTxt",true)	
                    << EAM(PostFix,"PostFix",true)	
                    << EAM(ByP,"ByP",true)	
                    << EAM(isCirc,"Circ",true,"In line mode if it's a loop (begin ~ end)")	
    );
    DoDevelopp(-1,aFullRes);

   std::string aRel = isCirc ? "NKS-Rel-ChantierCirculaire" : "NKS-Rel-ChantierLineaire";

    std::string aSFR =  BinPastis
                     +  aDir + std::string(" ") 
                     +  QUOTE(std::string(aRel + "@")+ aPat+ std::string("@")+ToString(aNbAdj)) + std::string(" ")
                     +  ToString(aFullRes) + std::string(" ")
                     +  StrMkT()
                     +  std::string("NbMinPtsExp=2 ")
                     +  NKS();


    std::cout << aSFR << "\n";
    System(aSFR,true);
    DoMkT();

    return 0;
}



int File(int argc,char ** argv)
{
    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAM(aFullDir)
                     <<EAM(aFullRes),
	LArgMain()  << EAM(ExpTxt,"ExpTxt",true)	
                    << EAM(PostFix,"PostFix",true)	
                    << EAM(ByP,"ByP",true)	
    );

    std::string aSFR =  BinPastis
                     +  aDir + std::string(" ") 
                     +  QUOTE(std::string("NKS-Rel-ByFile@")+  aPat) + std::string(" ")
                     +  ToString(aFullRes) + std::string(" ")
                     +  StrMkT()
                     +  std::string("NbMinPtsExp=2 ")
                     +  NKS();


    std::cout << aSFR << "\n";
    System(aSFR,true);
    DoMkT();
/*
*/

    return 0;
}









int main(int argc,char ** argv)
{
   MMD_InitArgcArgv(argc,argv);
   
/* =======  MODIF GREG, COMPREND PAS ?
	MMD_InitArgcArgv(argc,argv);
	aDir  = MMDir();
	
>>>>>>> .r799 
*/ 



//  APRES AVOIR SAUVEGARDER L'ARGUMENT DE TYPE ON LE SUPPRIME

    if (argc>=2)
    {
       TheType = argv[1];
       argv[1] = argv[0];
       argv++; argc--;
    }
	
    if (argc>=2)
    {
       aFullDir = argv[1];
       SplitDirAndFile(aDir,aPat,aFullDir);
    }
	
    aPatOri = aPat;

     StdAdapt2Crochet(aPat);
/*
    if ((aPat.find('@')!=(std::string::npos)) && (TheType != Type[3]))
    {
         aPat = "[["+aPat+"]]";
    }
*/

    MkFT= MMDir() + "MkTapioca";
    BinPastis = MMBin() + "Pastis ";
    ByP= MMNbProc();

    cTplValGesInit<std::string>  aTplFCND;
    anICNM = cInterfChantierNameManipulateur::StdAlloc(argc,argv,aDir,aTplFCND);

    if (TheType == Type[0])
    {
        return MultiECh(argc,argv);
    }
    else if (TheType == Type[1])
    {
        return All(argc,argv);
    }
    else if (TheType == Type[2])
    {
        return Line(argc,argv);
    }
    else if (TheType == Type[3])
    {
        return File(argc,argv);
    }

    std::cout << "TAPIOCA : Bad Value for networks configuration \n";
    std::cout << "Allowed enumerated values are  : \n";
    for (int aK=0 ; aK<aNbType ; aK++)
        std::cout << "    " << Type[aK] << "\n";
    std::cout << " ==================================\n";
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
