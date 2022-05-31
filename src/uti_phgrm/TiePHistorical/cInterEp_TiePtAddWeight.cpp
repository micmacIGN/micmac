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

#include "TiePHistorical.h"



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
aooter-MicMac-eLiSe-25/06/2007*/



void TiePtAddWeight(std::string aDir, std::string aInSH, std::string aOutSH, int nWeight, double dScaleL)
{
    if(dScaleL > 1 || dScaleL < 1)
        printf("The points in master images will be scaled by factor %.2lf\n", dScaleL);
    std::string aDir_inSH = aDir + "/Homol" + aInSH + "/";
    std::string aFileList = "FileList" + aInSH + ".txt";
    std::string cmmd = "find " + aDir_inSH + " > " + aFileList;
    cout<<cmmd<<endl;
    System(cmmd);

    std::string aDir_outSH = aDir + "/Homol" + aOutSH + "/";
    if (ELISE_fp::exist_file(aDir_outSH) == false)
        ELISE_fp::MkDir(aDir_outSH);

    std::vector<string> vFileList;

    std::string s;
    ifstream in1(aFileList);
    cout<<aFileList<<endl;
    while(getline(in1,s))
    {
        int nLen = s.length();
        if(nLen<4 || s.substr(nLen-3, nLen) != "txt")
            continue;

        vFileList.push_back(s);
    }

    int nTiePtNumTotal = 0;
    int nFileNum = 0;

    for(unsigned int i=0; i<vFileList.size(); i++)
    {
        std::string aFile = vFileList[i];
        ElPackHomologue aPackFull =  ElPackHomologue::FromFile(aFile);

        printf("%d: %s\n", i, aFile.c_str());

        int pos1 = aFile.rfind("/");
        int pos2 = aFile.substr(0,pos1).rfind("/");

        /*
        cout<<pos1<<endl;
        cout<<pos2<<endl;
        cout<<aFile.length()<<endl;
        cout<<aFile.substr(0,pos1)<<endl;
        cout<<aFile.substr(pos2,pos1-pos2)<<endl;
        */

        std::string aDirPastis = aDir_outSH + "/" + aFile.substr(pos2,pos1-pos2);
        if (ELISE_fp::exist_file(aDirPastis) == false)
            ELISE_fp::MkDir(aDirPastis);

        std::string aFileOut = aDirPastis + "/" + aFile.substr(pos1,aFile.length());
        int nTiePtNum = 0;
        FILE * fpOutput = fopen((aFileOut).c_str(), "w");
        for (ElPackHomologue::iterator itCpl=aPackFull.begin();itCpl!=aPackFull.end(); itCpl++)
        {
           ElCplePtsHomologues cple = itCpl->ToCple();
           Pt2dr p1 = cple.P1();
           Pt2dr p2 = cple.P2();

            fprintf(fpOutput, "%lf %lf %lf %lf %d\n", p1.x*dScaleL, p1.y*dScaleL, p2.x, p2.y, nWeight);
            nTiePtNum++;
        }
        fclose(fpOutput);
        printf("%d tie points in %s\n", nTiePtNum, aFile.c_str());
        nTiePtNumTotal += nTiePtNum;
        nFileNum++;

        std::string aImg1 = aFile.substr(pos2+7,pos1-pos2-7);
        std::string aImg2 = aFile.substr(pos1+1,aFile.length()-pos1-5);
        printf("mm3d SEL ./ %s %s KH=NT SH=%s SzW=[600,600]\n", aImg1.c_str(), aImg2.c_str(), aInSH.c_str());
    }
    printf("nTiePtNumTotal: %d; nTiePtNumHalf: %d;  nFileNum: %d\nResult saved in %s\n", nTiePtNumTotal, int(nTiePtNumTotal/2), nFileNum, aDir_outSH.c_str());
}

int TiePtAddWeight_main(int argc,char ** argv)
{
   cCommonAppliTiepHistorical aCAS3D;

   std::string aInSH = "";
   std::string aOutSH = "";

   int nWeight = 1;
   double dScaleL = 1;

   ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(nWeight,"Weight to be added"),
        LArgMain()
                    << aCAS3D.ArgBasic()
               << EAM(aInSH,"InSH",true,"Input Homologue extenion for NB/NT mode, Def=none")
               << EAM(aOutSH,"OutSH",true,"Output Homologue extenion for NB/NT mode, Def=InSH-WN (N means the weight)")
               << EAM(dScaleL,"ScaleL",true,"The factor used to scale the points in master images (for developpers only), Def=1")
               );

   if(aOutSH.length() == 0)
       aOutSH = aInSH + "-W" + std::to_string(nWeight);

   TiePtAddWeight(aCAS3D.mDir, aInSH, aOutSH, nWeight, dScaleL);

   return EXIT_SUCCESS;
}
