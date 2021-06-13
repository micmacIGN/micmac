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
#include "Apero.h"



/*******************************************/
/*                                         */
/*    cSolBasculeRig                        */
/*    cBasculementRigide                   */
/*                                         */
/*******************************************/



Fonc_Num GamCor(Fonc_Num aF,double aGamma)
{
   if (aGamma==1.0) return aF;
   if (aGamma>0) return 255.0 * pow(Max(0,aF)/255.0,1/aGamma);

   ELISE_ASSERT(false,"GamCor");
   return 0;
}


void cAppliApero::ExportRedressement(const cExportRedressement & anER,cPoseCam & aPC)
{
    const CamStenope * aCS = aPC.CurCam();
    {
       Pt3dr aD = aCS->DirK();
       double aTeta = acos(scal(aD,anER.DirTetaLim().Val()));
       if (aTeta > anER.TetaLimite())
          return;
    }
    std::string aName = aPC.Name();
    std::string aNameImOut= mOutputDirectory+mICNM->Assoc1To1(anER.KeyAssocOut(),aName,true);
    if (anER.DoOnlyIfNew().Val() && ELISE_fp::exist_file(aNameImOut))
       return;

    double aZSol = anER.ZSol().ValWithDef(aPC.AltiSol());
    double aResol = anER.Resol();
    Pt3dr aCenter = aCS->PseudoOpticalCenter();
    double aDZ = ElAbs(aCenter.z-aZSol);
    double aResolSol = aDZ / (aCS->Focale()*aCS->ScaleCamNorm());

// std::cout << "RES " << aResol << " SOL " << aResolSol << "\n";
    if (anER.ResolIsRel())
    {
       aResol *= aResolSol;
    }
    double aResolRel = aResol / aResolSol;

    std::string aNameImIn = anER.KeyAssocIn().IsInit()                           ?
                            mICNM->Assoc1To1(anER.KeyAssocIn().Val(),aName,true) :
                            aName                                                ;
    Tiff_Im aTIn = Tiff_Im::UnivConvStd(mDC+aNameImIn);
    double aScaleIm = anER.ScaleIm().Val();
    Pt2dr  aOffsIm = anER.OffsetIm().Val();
    
    // Calcul de la boite Terrain
    Pt2dr aSzIn = Pt2dr(aTIn.sz());
    Box2dr aBoxIn(Pt2dr(0,0),aSzIn);
    Pt2dr aCorner[4];
    aBoxIn.Corners(aCorner);
    Pt2dr aP0TerOut(1e30,1e30);
    Pt2dr aP1TerOut(-1e30,-1e30);
    for (int aK=0 ; aK<4 ; aK++)
    {
        Pt2dr aPIm = aCorner[aK];
        Pt2dr aPCam = aOffsIm + aPIm * aScaleIm;
        Pt3dr aPTer = aCS->F2AndZtoR3(aPCam,aZSol);
        Pt2dr aP2Ter(aPTer.x,aPTer.y);
        aP0TerOut.SetInf(aP2Ter);
        aP1TerOut.SetSup(aP2Ter);
    }
    Pt2di aIP0  = round_down(aP0TerOut/aResol);
    Pt2di aIP1  = round_up(aP1TerOut/aResol);
    aP0TerOut = Pt2dr(aIP0) * aResol;
    aP1TerOut = Pt2dr(aIP1) * aResol;

// std::cout << aP0TerOut << " "  << aIP0 << "\n";
// std::cout << aP1TerOut << " "  << aIP1 << "\n";
// std::cout << "========\n";

    Pt2di aSzOut = aIP1-aIP0;
    ELISE_fp::MkDirRec(aNameImOut);
 
    std::vector<Im2DGen *>   aVIn = aTIn.VecOfIm(round_ni(aSzIn));
    int aNbCh = (int)aVIn.size();
    ELISE_COPY(aTIn.all_pts(),aTIn.in(),StdOutput(aVIn));
    std::vector<Im2DGen *>   aVOut = aTIn.VecOfIm(aSzOut);

// std::cout << "Resol Rel " << aResolRel << " \n";
    cKernelInterpol1D * aKern = cKernelInterpol1D::StdInterpCHC(aResolRel,100);
/*
    delete aKern;
std::cout << "BBBBBBB\n";
getchar();
*/

    for (int anX=0; anX<aSzOut.x ; anX++)
    {
       for (int anY=0; anY<aSzOut.y ; anY++)
       {
            Pt2dr aPTer((anX+aIP0.x)*aResol,(anY+aIP0.y)*aResol);
            Pt2dr aPCam = aCS->R3toF2(Pt3dr(aPTer.x,aPTer.y,aZSol));
            Pt2dr aPIm  =  (aPCam-aOffsIm) / aScaleIm;
            for (int aK=0 ; aK<aNbCh ; aK++)
            {
                double aVal = aKern->Interpole(*(aVIn[aK]),aPIm.x,aPIm.y);
                aVOut[aK]->TronqueAndSet(Pt2di(anX,aSzOut.y -anY-1),aVal);
            }
       }
    }
    delete aKern;

    GenIm::type_el aTypeNOut =  aTIn.type_el();
    if(anER.TypeNum().IsInit())
      aTypeNOut = Xml2EL(anER.TypeNum().Val());


    Tiff_Im aTOut
            (
                  aNameImOut.c_str(),
                  aSzOut,
                  aTypeNOut,
                  Tiff_Im::No_Compr,
                  aTIn.phot_interp()
            );

   double anOfs = anER.Offset().Val();
   double aDyn  = anER.Dyn().Val();
   double aGamma  = anER.Gamma().Val();
   ELISE_COPY
   (
        aTOut.all_pts(),
        Tronque(aTypeNOut,GamCor((StdInPut(aVOut)-anOfs)*aDyn,aGamma)),
        aTOut.out()
   );

   if (anER.DoTFW().Val())
   {
       std::string  aNameTFW = StdPrefix(aNameImOut) + ".tfw";
       FILE * aFP = FopenNN(aNameTFW.c_str(),"w","cAppliApero::ExportRedressement");
       fprintf(aFP,"%lf %lf\n",aResol,0.0);
       fprintf(aFP,"%lf %lf\n",0.0,-aResol);
       fprintf(aFP,"%lf %lf\n",aP0TerOut.x,aP0TerOut.y + (aSzOut.y-1)*aResol);
       ElFclose(aFP);
   }

   DeleteAndClear(aVOut);
   DeleteAndClear(aVIn);
}


void cAppliApero::ExportRedressement(const cExportRedressement & anER)
{
    cElRegex * anAutom = anER.PatternSel().ValWithDef(0);

    for (int aKP=0 ; aKP<int(mVecPose.size()) ; aKP++)
    {
        cPoseCam * aPC = mVecPose[aKP];
        if ((anAutom==0) || (anAutom->Match(aPC->Name())))
        {
                 ExportRedressement(anER,*aPC);
        }
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
