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
#include "general/all.h"
#include "private/all.h"
#include <algorithm>



#define DEF_OFSET -12349876


struct cPGPS
{
    public :
       double mTime;
       Pt3dr  mP;
       int    mK;

       Pt2dr P2d() {return Pt2dr(mP.x,mP.y);}
    private :
      
};

void Init ( std::vector<cPGPS> & aVGps)
{

    cSystemeCoord aXmlSC = StdGetObjFromFile<cSystemeCoord>
                           (
                                "/home/binetr/Data/Muru/Radiale-Nord-2-22-10/SysCo-RTL-Muru.xml",
                                StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                "SystemeCoord",
                                "SystemeCoord"
                           );
    cSysCoord * aSCOut =  cSysCoord::FromXML(aXmlSC,(char *)0);
    cSysCoord *  aSCIn = cSysCoord::WGS84();
    ELISE_fp aFP
    (
         "/home/binetr/Data/Muru/Radiale-Nord-2-22-10/nav/TestGPS.txt",
          ELISE_fp::READ
    );
    char aBuf[1000];
    bool GotEOF = false;
    int aCpt=0;

    FILE * aFOut = FopenNN("/home/binetr/Data/Muru/Radiale-Nord-2-22-10/nav/AnalyseGps.txt","w","XXX");


    while (aFP.fgets(aBuf,1000,GotEOF) && (!GotEOF))
    {
         double aX,aY,aZ;
         cPGPS aPGps;

         int aNb = sscanf(aBuf,"%lf %lf %lf %lf",&aPGps.mTime,&aX,&aY,&aZ);
         ELISE_ASSERT(aNb==4,"sscanf");
         aX = ToRadian(aX,eUniteAngleDegre);
         aY = ToRadian(aY,eUniteAngleDegre);
   
         
         aPGps.mP = aSCOut->FromSys2This(*aSCIn,Pt3dr(aX,aY,aZ));
         aPGps.mK = aCpt;

         aVGps.push_back(aPGps);

         fprintf(aFOut,"%d %lf %lf %lf",aPGps.mK,aPGps.mP.x,aPGps.mP.y,aPGps.mP.z);
         if (aCpt>=2)
         {
            cPGPS aPrc = aVGps[aCpt-2];
            Pt3dr aVec = aPrc.mP - aPGps.mP;
            double ateta = atan2(aVec.y,aVec.x);
            fprintf(aFOut," %lf",ateta);
         }
         fprintf(aFOut,"\n");
 
         aCpt++;
    }
    aFP.close();
    ElFclose(aFOut);
}

//  Init("../Data/Muru/UTM/Tab-Appr_UTM.txt",aVAprIn,aVAprOut);
//  Init("../Data/Muru/UTM/Tab-Eval_UTM.txt",aVTestIn,aVTestOut);

//  bin/SysCoordPolyn /home/binetr/Data/Muru/UTM/Tab-Appr_UTM.txt mtoto.xml [4,4,1] [0,0,1] Test=/home/binetr/Data/Muru/UTM/Tab-Eval_UTM.txt 

int main(int argc,char ** argv)
{
   std::vector<cPGPS> aVGps;
   Init(aVGps);

   while (1)
   {
        int aK1,aK2;
        cin >> aK1 >> aK2;
        cPGPS aPG1 = aVGps[aK1];
        cPGPS aPG2 = aVGps[aK2];

        Pt2dr aP1 = aPG1.P2d();
        Pt2dr aP2 = aPG2.P2d();
        SegComp aSeg(aP1,aP2);

        double aDX = aSeg.abscisse(aP2)-aSeg.abscisse(aP1);
        double aDT = aPG2.mTime - aPG1.mTime;
        double aVMoy = aDX / aDT;

std::cout << "VM " << aVMoy << " " << aSeg.ordonnee(aP1) << " O2 " << aSeg.ordonnee(aP2) << "\n";
        for (int aK=aK1; aK<=aK2 ; aK++)
        {
            cPGPS aPG = aVGps[aK]; 
            Pt2dr aP = aPG.P2d();
         
            double aO = aSeg.ordonnee(aP);
            double aA = aSeg.abscisse(aP);
            double aDt = aPG.mTime -aPG1.mTime;
            double aEcA = aA - aVMoy * aDt;
            std::cout << "D=" << aO <<  " " << aA   << " EcA=" << aEcA << "\n";
        }

   }

/*
   std::vector<Pt3dr> aVAprIn,aVAprOut;
   std::vector<Pt3dr> aVTestIn,aVTestOut;
   std::string aNameAppr,aNameTest,aNameXML;

   Pt3di aDegXY,aDegZ;
   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAM(aNameAppr)
                    <<  EAM(aNameXML)
                    << EAM(aDegXY)
                    << EAM(aDegZ),
        LArgMain()  << EAM(aNameTest,"Test",true)
   );
*/
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
