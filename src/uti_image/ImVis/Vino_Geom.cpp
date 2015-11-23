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

#include "Vino.h"


#if (ELISE_X11)




/****************************************/
/*                                      */
/*          Grab Geom                   */
/*                                      */
/****************************************/

Pt2dr  cAppli_Vino::ToCoordAsc(const Pt2dr & aP)
{
   return Sup(Pt2dr(0,0),Inf(Pt2dr(SzW()),mScr->to_user(aP).mcbyc(mRatioFulXY)));
}


void cAppli_Vino::ShowAsc()
{
   mWAscH->clear();
   mWAscV->clear();
   Pt2dr aP00 =  ToCoordAsc(Pt2dr(0,0));
   Pt2dr aP10 =  ToCoordAsc((Pt2dr(SzW().x,0))) ; 
   Pt2dr aP01 =  ToCoordAsc(Pt2dr(0,SzW().y));  

   

   mWAscH->fill_rect
   (
       Pt2dr(aP00.x,0),
       Pt2dr(ElMax(aP00.x+1,aP10.x),LargAsc()),
       mWAscH->pdisc()(P8COL::yellow)
   );
   mWAscV->fill_rect
   (
         Pt2dr(0,aP00.y),
         Pt2dr(LargAsc(),ElMax(aP00.y+1,aP01.y)),
         mWAscV->pdisc()(P8COL::yellow)
  );

   std::string aStrZoom = "Zoom=" + StrNbChifSign(mScr->sc(),3); // ToString(mScr->sc()); 
   mW->fixed_string(Pt2dr(5,10),aStrZoom.c_str(),mW->pdisc()(P8COL::black),true);
}


void  cAppli_Vino::GUR_query_pointer(Clik aCl,bool)
{
    if (mModeGrab==eModeGrapZoomVino)
    {
         double aDY= aCl._pt.y - mP0Click.y;
         double aMulScale = pow(2.0,aDY/SpeedZoomGrab());
         // mScr->set(mTr0,mScale0*aMulScale,true);
         mScr->SetScArroundPW(mP0Click,mScale0*aMulScale,true);
         // std::cout << "GUR_query_pointer " << mP0Click << " " << aCl._pt << "\n";
    }
    if (mModeGrab==eModeGrapTranslateVino)
    {
       mScr->set(mTr0-(aCl._pt-mP0Click)/mScale0,mScale0,false);
    }
    if (mModeGrab==eModeGrapAscX)
    {
       mScr->set(mTr0+Pt2dr(aCl._pt.x-mP0Click.x,0)/mRatioFulXY.x,mScale0,false);
       ShowAsc();
    }
    if (mModeGrab==eModeGrapAscY)
    {
       mScr->set(mTr0+Pt2dr(0,aCl._pt.y-mP0Click.y)/mRatioFulXY.y,mScale0,false);
       ShowAsc();
    }

    if (mModeGrab==eModeGrapShowRadiom)
    {
         ShowOneVal(aCl._pt);
    }
}

void cAppli_Vino::ZoomMolette()
{
    double aSc =  mScale0 * pow(2.0,SpeedZoomMolette()*(mBut0==5?1:-1));
    mScr->SetScArroundPW(mP0Click,aSc,false);
}



void cAppli_Vino::ExeClikGeom(Clik aCl)
{

     if (mShift0)
     {
         if (mCtrl0)
         {
             mScr->set_max();
         }
         else
         {
              mModeGrab = eModeGrapZoomVino;
              mW->grab(*this);
              mScr->SetScArroundPW(mP0Click,mScr->sc(),false);
         }
     }
     else
     {
         if (mCtrl0)
         {
              mScr->SetScArroundPW(mP0Click,1.0,false);
         }
         else
         {
              mModeGrab = eModeGrapTranslateVino;
              mW->grab(*this);
         }
     }
     ShowAsc();
}

/****************************************/
/*                                      */
/*          STRING                      */
/*                                      */
/****************************************/

std::string StrNbChifSignNotSimple(double aVal,int aNbCh)
{
   if (aVal==1) return "1";
   if (aVal < 1)
   {
        if (aVal>0.1) return  ToString(aVal).substr(0,aNbCh+2);
        if (aVal>0.01) return  ToString(aVal).substr(0,aNbCh+3);

        double aLog10 = log(aVal) / log(10);
        int aLogDown =  round_down(ElAbs(aLog10));
        aVal = ElMin(1.0,aVal * pow(10,aLogDown));

        return ToString(aVal).substr(0,aNbCh+2) + "E-" + ToString(aLogDown);
        
       
   }

   if (aVal<100)
   {
       std::string aRes = ToString(aVal).substr(0,aNbCh+1);
       return aRes;
   }

   double aLog10 = log(aVal) / log(10);
   int aLogDown =  round_down(ElAbs(aLog10));
   aVal = ElMin(10.0,aVal / pow(10,aLogDown));
   return ToString(aVal).substr(0,aNbCh+2) +  "E" + ToString(aLogDown);
}

std::string StrNbChifSign(double aVal,int aNbCh)
{
    return SimplString(StrNbChifSignNotSimple(aVal,aNbCh));
}

std::string SimplString(std::string aStr)
{
   if (aStr.find('.') == std::string::npos)
      return aStr;
   int aK= aStr.size()-1;
   while ((aK>0) && (aStr[aK]=='0'))
     aK--;
   if (aStr[aK]=='.') 
     aK--;
   aK++;
   return aStr.substr(0,aK);
}




#endif



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
