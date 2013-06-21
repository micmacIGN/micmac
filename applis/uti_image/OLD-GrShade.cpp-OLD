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

Im2DGen AllocImGen(Pt2di aSz,const std::string & aName)
{
    return D2alloc_im2d(type_im(aName),aSz.x,aSz.y);
}


//   aVPdsFiltre :
//
//    0 Ciel Visible
//    1 Local
//    2  IgnE
//    3 Median
//



int main(int argc,char ** argv)
{
     std::string aNameIn;
     std::string aNameOut;
     std::string aNameCol="";
     Pt2di aP0Glob(0,0),aSzGlob(0,0);
     INT aNbDir = 20;
     REAL aFZ = 1.0;

     REAL aPdsAnis = 0.95;
     INT  aBrd = -1;
     std::string aTMNt = "real4";
     std::string aTShade = "real4";
     INT aDequant =0;
     INT aVisu = 0;
     REAL aHypsoDyn = -1.0;
     REAL aHypsoSat = 0.5;

     Pt2di aSzMaxDalles (3000,3000);
     INT aSzRecDalles = 300;
     std::string aModeOmbre="CielVu";
     std::string aFileMasq="";

     double  aDericheFact=2.0;
     int     aNbIterF = 4;
     double  aFactExp = 0.95;
     double aDyn = 1.0;
     int aNbMed = 100;
     int aNbIterMed = 1;

     Tiff_Im::SetDefTileFile(1<<15);

     std::vector<double> aVPdsFiltre;


     std::string aModeColor = "IntensShade";

     double aTetaH = 25.0;
     double anAzimut = 0.0;
     double aDynMed = 1.0;

     ElInitArgMain
     (
           argc,argv,
           LArgMain() << EAM(aNameIn) ,
           LArgMain() << EAM(aNameOut,"Out",true)
	              << EAM(aNameCol,"FileCol",true)
                      << EAM(aVisu,"Visu",true)
                      << EAM(aP0Glob,"P0",true)
                      << EAM(aSzGlob,"Sz",true)
                      << EAM(aFZ,"FZ",true)
                      << EAM(aDynMed,"DynMed",true)
                      << EAM(aPdsAnis,"Anisotropie",true)
		      << EAM(aNbDir,"NbDir",true)
		      << EAM(aBrd,"Brd",true)
                      << EAM(aTMNt,"TypeMnt",true)
                      << EAM(aTShade,"TypeShade",true)
                      << EAM(aDequant,"Dequant",true)
                      << EAM(aHypsoDyn,"HypsoDyn",true)
                      << EAM(aHypsoSat,"HypsoSat",true)
		      << EAM(aSzMaxDalles,"SzMaxDalles",true)
		      << EAM(aSzRecDalles,"SzRecDalles",true)
		      << EAM(aModeOmbre,"ModeOmbre",true)
		      << EAM(aFileMasq,"Mask",true)
		      << EAM(aDericheFact,"DericheFact",true)
		      << EAM(aNbIterF,"NbIterF",true)
		      << EAM(aFactExp,"FactExp",true)
		      << EAM(aDyn,"Dyn",true)
                      << EAM(aVPdsFiltre,"PdsF",true,"[CielVu,Local,Grad,Med]")
                      << EAM(aModeColor,"ModeColor",true)
                      << EAM(aNbMed,"NbMed",true)
                      << EAM(aNbIterMed,"NbIterMed",true)
                      << EAM(aTetaH,"TetaH",true)
                      << EAM(anAzimut,"Azimut",true)
    );


    double aPdsDef = aVPdsFiltre.size() ? 0 : 1;
    for (int aK=aVPdsFiltre.size() ; aK<4 ; aK++)
       aVPdsFiltre.push_back(aPdsDef);

    double aSPdsF = 0;
    for (int aK=0 ; aK<4 ; aK++)
       aSPdsF += aVPdsFiltre[aK];
    for (int aK=0 ; aK<4 ; aK++)
        aVPdsFiltre[aK] /= aSPdsF;
       
 
    std::string aDir,aNameFileIn;
    SplitDirAndFile(aDir,aNameFileIn,aNameIn);



     bool WithHypso = (aHypsoDyn>0) || (aNameCol != "");
     // bool WithCol =   (aNameCol != "");


    if (aNameOut=="")
       aNameOut = StdPrefix(aNameIn) +std::string("Shade.tif");

     Tiff_Im aFileIn = Tiff_Im::StdConvGen(aNameIn,1,true,false);
     if (aSzGlob== Pt2di(0,0))
        aSzGlob = aFileIn.sz() -aP0Glob;
     Fonc_Num aFIn = aFileIn.in_gen(Tiff_Im::eModeCoulGray,Tiff_Im::eModeNoProl);

    {
        Tiff_Im
        (
             aNameOut.c_str(),
             aSzGlob,
             GenIm::u_int1,
	     Tiff_Im::No_Compr,
	     WithHypso  ? Tiff_Im::RGB : Tiff_Im::BlackIsZero
        );
    }
    Tiff_Im aTifOut(aNameOut.c_str());

     if (aSzMaxDalles.x<0) aSzMaxDalles = aSzGlob;
     Pt2di aPRD(aSzRecDalles,aSzRecDalles);
     cDecoupageInterv2D aDecoup
	                (
                            Box2di(aP0Glob,aP0Glob+aSzGlob),
			    aSzMaxDalles,
			    Box2di(-aPRD,aPRD)
			);

     Im2DGen aMnt =    AllocImGen(aDecoup.SzMaxIn(),aTMNt);
     Im2DGen aShade =  AllocImGen(aDecoup.SzMaxIn(),aTShade);

     cout << "SZ Max In " << aDecoup.SzMaxIn() << endl;
     REAL aRatio = ElMin(800.0/aSzGlob.x,700.0/aSzGlob.y);
     Video_Win * pW  = aVisu                          ?
                       Video_Win::PtrWStd(Pt2di(Pt2dr(aSzGlob)*aRatio)) :
                       0                              ;

     aTetaH *= (2*PI)/360.0;
     anAzimut *= (2*PI)/360.0;

     for (int aKDec=0; aKDec<aDecoup.NbInterv() ; aKDec++)
     {

         Box2di aBoxIn = aDecoup.KthIntervIn(aKDec);
	 Pt2di aSzIn = aBoxIn.sz();
	 Pt2di aP0In = aBoxIn.P0();

	 cout << "DEQ " << aDequant << "Sz In " << aSzIn <<endl;

         REAL aVMin;
         if (aDequant)
         {
             ElImplemDequantifier aDeq(aSzIn);
             aDeq.SetTraitSpecialCuv(true);
             aDeq.DoDequantif(aSzIn, trans(aFIn,aP0In),true);
	     REAL aVMax;
             ELISE_COPY
             (
                  rectangle(Pt2di(0,0),aSzIn),
                  aDeq.ImDeqReelle() * aFZ,
                  aMnt.out() | VMax(aVMax) |VMin(aVMin)
             );
	      
         }
         else 
	 {
             ELISE_COPY
             (
                  rectangle(Pt2di(0,0),aSzIn),
                  trans(aFIn,aP0In)*aFZ,
                  aMnt.out()|VMin(aVMin)
             );
	 }
         Im2D_Bits<1> aIMasq(aSzIn.x,aSzIn.y,1);

	 if (aFileMasq!="")
	 {
	     double aDif=100;
	     Tiff_Im aFM = Tiff_Im::StdConvGen(aDir+aFileMasq,1,true,false);
             ELISE_COPY
             (
                  select(rectangle(Pt2di(0,0),aSzIn),trans(!aFM.in_proj(),aP0In)),
		  aVMin-aDif,
                     aMnt.out()
                  |  (aIMasq.out() << 0)
             );
	     aVMin-= aDif;
	 }

         if (aBrd>0)
         {
            cout << "VMin = " << aVMin <<endl;
            ELISE_COPY(aMnt.border(aBrd),aVMin-1000,aMnt.out());
         }

     // Im2D_REAL4 aShade(aSzGlob.x,aSzGlob.y);
         ELISE_COPY(aShade.all_pts(),0,aShade.out());

         if (pW)
            pW = pW->PtrChc(Pt2dr(aP0Glob-aP0In),Pt2dr(aRatio,aRatio));

         REAL SPds = 0;
         REAL aSTot = 0;
         REAL Dyn = 1.0;
         if (aTShade != "u_int1")
            Dyn = 100;

         bool Done = false;
         if (   (aModeOmbre=="CielVu") 
             || ((aModeOmbre=="Mixte") && (aVPdsFiltre[0] > 0.0))
            )
	 {
            std::cout << "BEGIN CIEL" << endl;
            Done = true;
            for (int aK=0 ; aK< 2 ; aK++)
            {
               SPds = 0;
               for (int i=0; i<aNbDir; i++)
               {
                  REAL Teta  = (2*PI*i) / aNbDir ;
                  Pt2dr U(cos(Teta),sin(Teta));
                  Pt2di aDir = Pt2di(U * (aNbDir * 4));
                  REAL Pds = (1-aPdsAnis) + aPdsAnis *ElSquare(1.0 - euclid(U,Pt2dr(0,1))/2);
                  if (aK==1)
                     Pds = (Pds*Dyn) / (2*aSTot);
                  Symb_FNum Gr = (1-cos(PI/2-atan(gray_level_shading(aMnt.in()))))
                             *255.0;
                  cout << "Dir " << i << " Sur " << aNbDir <<  " P= " << Pds << endl;
                  SPds  += Pds;
                  if (aK==1)
                  {
	             ELISE_COPY
	             (
	                 line_map_rect(aDir,Pt2di(0,0),aSzIn),
	                 Min(255*Dyn,aShade.in()+Pds*Gr),
	                   aShade.out() 
                         // | (pW ? (pW->ogray()<<(aShade.in()/SPds)) : Output::onul())
                         | (pW ? (pW->ogray()<<(Gr)) : Output::onul())
                     );
                  }
               }
               aSTot  = SPds;
            }
            double aMul = (aModeOmbre=="Mixte") ? aVPdsFiltre[0] : 1.0;
            ELISE_COPY(aShade.all_pts(),aShade.in()*(aMul/SPds),aShade.out());
            SPds = aMul;
            std::cout << "BEGIN CIEL" << endl;
	 }
	 if (
                      (aModeOmbre=="Local")
                   || ((aModeOmbre=="Mixte") && (aVPdsFiltre[1] > 0.0))
                 )
	 {
               std::cout << "BEGIN LOCAL" << endl;
               Done = true;
               Fonc_Num aFonc = aMnt.in_proj();
               Fonc_Num aMoy = aFonc;
               for (int aK=0 ; aK<aNbIterF; aK++)
                   aMoy =    canny_exp_filt(aMoy*aIMasq.in_proj(),aFactExp,aFactExp) 
                          /  Max(0.1,canny_exp_filt(aIMasq.in_proj(),aFactExp,aFactExp));

               double aMul = (aModeOmbre=="Mixte") ? aVPdsFiltre[1] : 1.0;
               ELISE_COPY
               (
	          rectangle(Pt2di(0,0),aSzIn),
		  Max(0,Min(255, aShade.in() +(128+(aFonc-aMoy)*aDyn)* aMul)),
		  aShade.out()
               );
               SPds += aMul;
               std::cout << "END LOCAL" << endl;
	 }
	 if (
                      (aModeOmbre=="Med")
                   || ((aModeOmbre=="Mixte") && (aVPdsFiltre[3] > 0.0))
                 )
	 {
              std::cout << "BEGIN MED" << endl;

               Done = true;
               Fonc_Num aFonc = round_ni(aMnt.in_proj()*aDynMed);
               int aVMax,aVMin;

               ELISE_COPY
               (
                   rectangle(Pt2di(-1,-1),aSzIn+Pt2di(1,1)),
                   aFonc,
                   VMin(aVMin)|VMax(aVMax)
               );

               Fonc_Num aMoy = aFonc-aVMin;

               for (int aK=0 ; aK<aNbIterMed; aK++)
                   aMoy =    rect_median(aMoy,aNbMed,aVMax-aVMin+1); 

               aMoy = aMoy + aVMin;

               double aMul = (aModeOmbre=="Mixte") ? aVPdsFiltre[3] : 1.0;
               ELISE_COPY
               (
	          rectangle(Pt2di(0,0),aSzIn),
		  Max(0,Min(255, aShade.in() +(128+((aFonc-aMoy)*aDyn)/aDynMed)* aMul)),
		  aShade.out()
               );
               SPds += aMul;
              std::cout << "END MED" << endl;
	 }
	 if (
                      (aModeOmbre=="IgnE")
                   || ((aModeOmbre=="Mixte") && (aVPdsFiltre[2] > 0.0))
                 )
	 {
int aCpt=0; aCpt++;
std::cout << "IGN E " << aCpt << " " << aKDec << "\n";
             Done = true;
if (aCpt>0)
{
	     Symb_FNum aGrad =  deriche(aMnt.in_proj(),aDericheFact);
             Symb_FNum aGx = (aGrad.v0());
             Symb_FNum aGy = (aGrad.v1());
	     Symb_FNum aNG = sqrt(1+Square(aGx)+Square(aGy));

	     Symb_FNum aNx (aGx/aNG);
	     Symb_FNum aNy (aGy/aNG);
	     Symb_FNum aNz (1/aNG);



	     Pt2dr  aDirS = Pt2dr::FromPolar(1.0,anAzimut) * Pt2dr(1,0);
	     
	     double aSx = aDirS.x * sin(aTetaH);
	     double aSy = aDirS.y * sin(aTetaH);
	     double aSz = cos(aTetaH);

	     Symb_FNum aScal(aNx*aSx+aNy*aSy+aNz*aSz);

std::cout << "AAAAAAAaa" << endl;
             double aMul = (aModeOmbre=="Mixte") ? aVPdsFiltre[2] : 1.0;
             ELISE_COPY
             (
	          rectangle(Pt2di(0,0),aSzIn),
		  Max(0,aShade.in() + 255*aScal * aMul),
		  aShade.out()
             );
             SPds += aMul;
std::cout << "BBBBbbb" << endl;
}
	 }
	 if (! Done)
	 {
	     ELISE_ASSERT(false,"Unknown ModeOmbre");
	 }



        Fonc_Num aFoncRes = Max(0,Min(255,aShade.in()/SPds));
        if (WithHypso)
        {
            Fonc_Num  aFIntens = aFoncRes;
            Fonc_Num  aFTeinte = trans(aFIn,aP0In)*aHypsoDyn;
            Fonc_Num  aFSat = 255*aHypsoSat;

            if (aNameCol!="")
            {
                Tiff_Im aFileCol = Tiff_Im::StdConvGen(aDir+aNameCol,-1,true,false);
	        Symb_FNum aFNC(trans(rgb_to_its(aFileCol.in()),aP0In));
               
                if (aModeColor == "IntensShade")
                {
                    aFIntens = aFoncRes;
                    aFTeinte = aFNC.v1();
                    aFSat = aFNC.v2();
                }
                else if (aModeColor == "BackRGB")
                {
                   aFIntens = aIMasq.in()*aFoncRes + (1- aIMasq.in()) * aFNC.v0();
                   aFTeinte = aFNC.v1();
                   aFSat = aFNC.v2() * (1- aIMasq.in());
                }
                else if (aModeColor == "GrayBackRGB")
                {
                   aFIntens = aIMasq.in()*aFoncRes + (1- aIMasq.in()) * aFNC.v0();
                   aFTeinte = aFNC.v1();
                   aFSat = aFNC.v2()*(1-aIMasq.in());
                }
                else 
                {
                    ELISE_ASSERT(false,"Unknown mode color");
                }
            }
            aFoncRes = its_to_rgb(Virgule(aFIntens,aFTeinte,aFSat));
            //aFoncRes = its_to_rgb(Virgule(aFoncRes,trans(aFIn,aP0In)*aHypsoDyn,255*aHypsoSat));
        }
/*
	if (WithCol)
	{
            Tiff_Im aFileCol(aNameCol.c_str());
	    Symb_FNum aFNC(trans(rgb_to_its(aFileCol.in()),aP0In));
	    aFoncRes = its_to_rgb(Virgule(aFoncRes,aFNC.v1(),aFNC.v2()*aHypsoSat));
	    // aFoncRes = aFileCol.in();
	}
*/

     // Tiff_Im::Create8BFromFonc(aNameOut,aShade.sz(),aShade.in()/SPds);


	cout << "WithHypso " << WithHypso << " DIM " << aFoncRes.dimf_out() <<  endl;
        Box2di aBoxOut = aDecoup.KthIntervOut(aKDec);
        ELISE_COPY
        (
            rectangle(aBoxOut.P0()-aP0Glob,aBoxOut.P1()-aP0Glob),
	    trans(aFoncRes,aP0Glob-aP0In),
	    aTifOut.out()
        );
     }


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
