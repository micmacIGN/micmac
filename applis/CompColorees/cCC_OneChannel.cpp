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

#include "CompColore.h"


/*************************************************/
/*                                               */
/*          cCC_OneChannel                       */
/*                                               */
/*************************************************/

Pt2di P0_CornerMort(const cImageCmpCol & aICC)
{
    return aICC.BoxPixMort().IsInit() ?
           aICC.BoxPixMort().Val().HautG() :
           Pt2di(0,0);
}
Pt2di P1_CornerMort(const cImageCmpCol & aICC)
{
    return aICC.BoxPixMort().IsInit() ?
           aICC.BoxPixMort().Val().BasD() :
           Pt2di(0,0);
}




cCC_OneChannel::cCC_OneChannel
(
   double                aScale,
   const cImageCmpCol &  anICC,
   cCC_Appli &           anAppli,
   cCC_OneChannel *      aMaster
) :
    mICC      (anICC),
    mAppli    (anAppli),
    mDir      (mAppli.WorkDir()),
    mNameInit (anICC.NameOrKey()),
    mNameCorrige (  (aMaster!=0)   ?
                    anAppli.ICNM()->StdCorrect(mNameInit,aMaster->mNameGeomCorrige,true) :
		    mNameInit
                 ),
    mNameGeomCorrige (
                          anICC.KeyCalcNameImOfGeom().IsInit()                                      ?
			  anAppli.ICNM()->Assoc1To1(anICC.KeyCalcNameImOfGeom().Val(),mNameCorrige,true) :
                          mNameCorrige
                     ),
    mNameFile (mAppli.WorkDir() + mNameCorrige),
    mFileIm   (Tiff_Im::UnivConvStd(mNameFile)),
    mTypeIn   (
                   anICC.TypeTmpIn().IsInit() ?
                   Xml2EL(anICC.TypeTmpIn().Val())    :
                   mFileIm.type_el()
              ),
    mP0GlobIn     (P0_CornerMort(anICC)),
    mP1GlobIn     (mFileIm.sz()-P1_CornerMort(anICC)),
    mCam      ( CamCompatible_doublegrid(mDir+anAppli.ICNM()->Assoc1To1(anAppli.CCC().KeyCalcNameCalib(),mNameGeomCorrige,true))),
    // mCam      ( Std_Cal_From_File(mDir+anAppli.ICNM()->Assoc1To1(anAppli.CCC().KeyCalcNameCalib(),mNameGeomCorrige,true))),
    // mCam      ( Std_Cal_From_File(mDir+mICC.CalibCam())),
    mGrid     (0),
    mOriFus   (0.0,0.0),
    mScaleF   (aScale),
    mChInMade (false),
    mMaxChOut (-1)
{

     Pt2dr aMil = Pt2dr(mCam->Sz()) / 2.0;
     Pt2dr aP1 = aMil + Pt2dr(1,0);
     Pt2dr aP2 = aMil - Pt2dr(1,0);
     Pt2dr aQ1 = mCam->F2toC2(aP1);
     Pt2dr aQ2 = mCam->F2toC2(aP2);
     double aDist = euclid(aQ1,aQ2)/2.0;
     mFactF2C2 = 1/aDist;

     std::cout << "================ Fact " << mFactF2C2 << "\n";
}

Pt2di cCC_OneChannel::SzCurInput() const
{
   return mCurP1In-mCurP0In;
}

CamStenope & cCC_OneChannel::Cam()
{
   return *mCam;
}


Box2dr cCC_OneChannel::BoxIm2BoxFus(const Box2dr & aBox) 
{
   return  ImageOfBox(aBox,50);
}

Box2dr cCC_OneChannel::GlobBoxFus() 
{
    return BoxIm2BoxFus(Box2dr(mP0GlobIn,mP1GlobIn));
}

cCC_OneChannel::~cCC_OneChannel()
{
   delete mGrid;
   DeleteAndClear(mChIns);
}



Pt2dr cCC_OneChannel::Direct(Pt2dr aP) const
{
    aP = aP;
    return (mAppli.CCC().CorDist().Val()?mCam->DistInverse(aP):aP) * mScaleF;
}

bool cCC_OneChannel::OwnInverse(Pt2dr & aP) const 
{
    aP = aP/mScaleF;
    aP = mAppli.CCC().CorDist().Val()?mCam->DistDirecte(aP):aP;
    aP= aP;
    return true;
}

bool  cCC_OneChannel::SetBoxOut(const Box2di & aBox) 
{
    Pt2di aPRin(aRabInput,aRabInput);

    mOriFus = Pt2dr(aBox._p0);
    Box2dr aBoxI = ImageRecOfBox(I2R(aBox),50);

    mCurP0In = Sup(mP0GlobIn,round_down(aBoxI._p0)-aPRin);
    mCurP1In = Inf(mP1GlobIn,round_up(aBoxI._p1)+aPRin);

    if ((mCurP0In.x>= mCurP1In.x) || (mCurP0In.y>= mCurP1In.y))
       return false;
    return true;
}

void cCC_OneChannel::VirtualPostInit()
{
    mGrid = new cDbleGrid
                (
	            true,
	            Pt2dr(mP0GlobIn),
	            Pt2dr(mP1GlobIn),
		    Pt2dr(mAppli.CCC().StepGrid(), mAppli.CCC().StepGrid()),
		    *this
	        );
}






Pt2dr  cCC_OneChannel::ToFusionSpace(const Pt2dr & aP) const
{
   // return (mGrid->Direct(aP+mP0In) - mOriFus) * mScale;
   return (mGrid->Direct(aP+Pt2dr(mCurP0In)) - mOriFus) ;
}


Pt2dr  cCC_OneChannel::FromFusionSpace(const Pt2dr & aP) const
{
   // return mGrid->Inverse(aP/mScale + mOriFus) -mP0In;
   return mGrid->Inverse((aP+mOriFus) ) -Pt2dr(mCurP0In);
}


int cCC_OneChannel::MaxChOut() 
{
   return mMaxChOut;
}

const std::string &   cCC_OneChannel::NameCorrige() const
{
   return mNameCorrige;
}

const std::string &   cCC_OneChannel::NameGeomCorrige() const
{
   return mNameGeomCorrige;
}

void cCC_OneChannel::DoChIn()
{

  Output aOutGlob = Output::onul();



  Fonc_Num aFFRef=  0;

  int aCptChIn=0;
  for (int aK=0 ; aK<mFileIm.in().dimf_out() ; aK++)
  {
      Output aOutLoc = Output::onul();
      bool First = true;
      for 
      (
          std::list<cChannelCmpCol>::const_iterator  itCCC = mICC.ChannelCmpCol().begin();
          itCCC != mICC.ChannelCmpCol().end();
	  itCCC++
      )
      {
          if (itCCC->In() == aK)
	  {
	      ElSetMax(mMaxChOut,itCCC->Out());
              if (! mChInMade)
              {
	         mChIns.push_back
                 (
                     new cChannelIn
                         (
                            mAppli,
                            mTypeIn,
                            *this,
                            SzCurInput(),
                            *itCCC
                         )
                 );
              }
              else
                 mChIns[aCptChIn]->Resize(SzCurInput());
	      if (First)
	          aOutLoc = mChIns[aCptChIn]->OutForInit();
              else
	          aOutLoc = aOutLoc | mChIns[aCptChIn]->OutForInit();
              First = false;
              aCptChIn++;
	  }
      }
      if (mICC.FlattField().IsInit())
      {
           const cFlattField & aFF = mICC.FlattField().Val();
           double aVRef = aFF.RefValue()[ElMin(aK,int(aFF.RefValue().size()-1))];
           if (aK==0)
              aFFRef = aVRef;
           else
              aFFRef = Virgule(aFFRef,aVRef);
      }
      if (aK==0)
         aOutGlob = aOutLoc;
      else
         aOutGlob = Virgule(aOutGlob,aOutLoc);
  }

  Fonc_Num aFMult = 1;
  if (mICC.FlattField().IsInit())
  {
     const cFlattField & aFF = mICC.FlattField().Val();
     aFMult = aFFRef / Tiff_Im::BasicConvStd(mAppli.WorkDir()+aFF.NameFile()).in_proj();
  }


  Fonc_Num aFin = mFileIm.in_proj();
  for (int aK=0 ; aK<mICC.NbFilter().Val() ; aK++)
  {
      int aSzF = mICC.SzFilter().Val();
      aFin = rect_som(aFin,aSzF) /ElSquare(1+2*aSzF);
  }

// std::cout << mICC.NbFilter().Val() << " " << mNameFile << "\n";
// Video_Win aW = Video_Win::WStd(Pt2di(800,800),1);
  ELISE_COPY
  (
      rectangle(Pt2di(0,0),SzCurInput()),
      trans(aFin*aFMult,mCurP0In),
      aOutGlob 
  );

  mChInMade = true;
// getchar();
}

const  std::vector<cChannelIn *> & cCC_OneChannel::ChIn()
{
   return mChIns;
}


/*************************************************/
/*                                               */
/*          cCC_OneChannelSec                    */
/*                                               */
/*************************************************/


cCC_OneChannelSec::cCC_OneChannelSec
(
     double             aScale,
     const cImSec &     anIS,
     cCC_OneChannel &   aMaster,
     cCC_Appli &        anAppli
)  :
    cCC_OneChannel    (aScale,anIS.Im(),anAppli,&aMaster),
    mIS               (anIS),
    mNameCorresp      (   mIS.DirCalcCorrep().Val()
                        + anAppli.ICNM()->NamePackWithAutoSym
                                       (
				              mIS.KeyCalcNameCorresp(),
					      aMaster.NameGeomCorrige(),
					      NameGeomCorrige()
                                       )
                      ),
    mMaster           (aMaster),
    mH_M2This         (cElHomographie::Id()),
    mH_This2M         (cElHomographie::Id()),
    mGM2This          (0)
{
   bool ByGrid = false;
   if (StdPostfix(mNameCorresp)=="xml")
   {
       cElXMLTree  aTree (mDir + mNameCorresp);
       ByGrid = (aTree.Get("GridDirecteEtInverse") != 0);
   }

   if (ByGrid)
   {
       cDbleGrid::cXMLMode aXmlMode;
       mGM2This = new cDbleGrid(aXmlMode,mDir,mNameCorresp);
   }
   else 
   {

      ElPackHomologue aPackInit =  ElPackHomologue::FromFile(mDir + mNameCorresp);

      if (mIS.OffsetPt().IsInit())
      {
         Pt2dr anOfs = mIS.OffsetPt().Val();
         for 
         (
                ElPackHomologue::iterator itPt = aPackInit.begin();
                itPt != aPackInit.end();
                itPt++
          )
          {
              itPt->P1() =  itPt->P1()+anOfs;
              itPt->P2() =  itPt->P2()+anOfs;
          }
      }

      ElPackHomologue aPackCorDist;
      for 
      (
             ElPackHomologue::const_iterator itP = aPackInit.begin();
             itP != aPackInit.end();
	     itP++
      )
      {
/*
std::cout  << "PTS : " << itP->P1() << itP->P2() << "\n";
std::cout  << "DIR : " <<  mMaster.Cam().DistDirecte(itP->P1()) <<  mCam->DistDirecte(itP->P2()) << "\n";
std::cout  << "INV : " <<  mMaster.Cam().DistInverse(itP->P1()) <<  mCam->DistInverse(itP->P2()) << "\n";
getchar();
*/
              aPackCorDist.Cple_Add
	      (
                  ElCplePtsHomologues
                  (
	              mMaster.Cam().DistInverse(itP->P1()),
		      mCam->DistInverse(itP->P2())
/*
 mMaster.Cam().DistInverse( mMaster.Cam().DistDirecte(itP->P1())),
 mCam->DistInverse( mCam->DistDirecte(itP->P1()))
*/
                  )
	      );
       }

       if (mIS.L2EstimH().Val())
            mH_M2This = cElHomographie(aPackCorDist,true);
       else if (mIS.L1EstimH().Val())
            mH_M2This = cElHomographie(aPackCorDist,false);
       else
            mH_M2This = cElHomographie::RansacInitH
                         (
                              aPackCorDist,
                              mIS.NbTestRansacEstimH().Val(),
                              mIS.NbPtsRansacEstimH().Val()
                         );



         for 
         (
             std::list<Pt2dr>::const_iterator itP=mIS.PonderaL2Iter().begin();
             itP!=mIS.PonderaL2Iter().end();
             itP++
         )
         {
              double aMax = itP->x;
              double aPond = itP->y;
              int aNb=0;
              int aNbOK=0;
              double aSomDist=0;
              for 
              (
                  ElPackHomologue::iterator itCpl=aPackCorDist.begin();
                  itCpl!=aPackCorDist.end();
                  itCpl++
              )
              {
                   Pt2dr aP1 = itCpl->P1();
                   Pt2dr aP2 = itCpl->P2();
                   Pt2dr aQ2 = mH_M2This.Direct(aP1);
                   double aDist = euclid(aP2,aQ2) * mFactF2C2;
                   double aPds  = (aDist>aMax)                    ? 
                                  0                               : 
                                  1/sqrt(1+ElSquare(aDist/aPond)) ;
                   itCpl->Pds() = aPds;
                   aNb++;
                   if (aDist<aMax)
                   {
                        aNbOK++;
                        aSomDist+=aDist;
                   }
              }
              mH_M2This = cElHomographie(aPackCorDist,true);
              std::cout << "OK = " << aNbOK << " on " << aNb  << " DMoy " << (aSomDist/aNbOK) << "\n";
         }

         mH_This2M = mH_M2This.Inverse();
    }
}

Pt2dr cCC_OneChannelSec::Direct(Pt2dr aP) const
{
    if (mGM2This)
    {
       aP = mGM2This->Inverse(aP);
       return (mAppli.CCC().CorDist().Val()? mMaster.Cam().DistInverse(aP) : aP) * mScaleF;
    }
    aP = mH_This2M.Direct(mCam->DistInverse(aP));
    return (mAppli.CCC().CorDist().Val()? aP : mMaster.Cam().DistDirecte(aP)) * mScaleF;
}

bool cCC_OneChannelSec::OwnInverse(Pt2dr & aP) const 
{
    aP = aP/mScaleF;
    if (mGM2This)
    {
         aP =  mAppli.CCC().CorDist().Val()? mMaster.Cam().DistDirecte(aP) : aP;
	 aP = mGM2This->Direct(aP);
    }
    else
    {

       aP =   mAppli.CCC().CorDist().Val()?aP :mMaster.Cam().DistInverse(aP);
       aP =   mCam->DistDirecte(mH_M2This.Direct(aP));
    }


    return true;
}

cCC_OneChannelSec::~cCC_OneChannelSec()
{
   delete mGM2This;
}

void cCC_OneChannelSec::TestVerif()
{
   if (!mIS.VerifHoms().IsInit())
      return;

    const cVerifHoms &  aVH = mIS.VerifHoms().Val();
    double aSD = 0.0;
    double aSDV = 0.0;
    double aS1 = 0.0;

    double aDZ =0,anExag=0;
    El_Window * aW=0;
    Bitm_Win * aBW = 0;

    bool aVXY=false;
    double anExagXY=0;
    Im2D_REAL4 aVX(1,1);
    Im2D_REAL4 aVY(1,1);

    

    std::string aNameFile;


    if (aVH.VisuEcart().IsInit())
    {
       cVisuEcart aVE = aVH.VisuEcart().Val();
       Pt2dr aSz = Pt2dr(mFileIm.sz());
       aDZ = aVE.SzW() / ElMax(aSz.x,aSz.y);
       if (aVE.NameFile().IsInit())
       {
          aBW = new Bitm_Win("Toto.tif",GlobPal(),Pt2di(aSz*aDZ));
          ELISE_COPY(aBW->all_pts(),P8COL::black,aBW->odisc());
          aW = aBW;
          aNameFile = mAppli.WorkDir() + aVE.NameFile().Val();
          // aW = new Bitm_Win("Toto.tif",aSz*aDZ);
       }
       else
          aW =  Video_Win::PtrWStd(Pt2di(aSz*aDZ));
       anExag = aVE.Exag();
       if (aVE.Images2Verif().IsInit())
       {
           cImages2Verif aCI2V = aVE.Images2Verif().Val();
	   aVX = Im2D_REAL4::FromFileStd(mDir+aCI2V.X());
	   aVY = Im2D_REAL4::FromFileStd(mDir+aCI2V.Y());
	    anExagXY =  aCI2V.ExagXY();
	   aVXY = true;
       }
    }
    TIm2D<REAL4,REAL8> aTX(aVX);
    TIm2D<REAL4,REAL8> aTY(aVY);

      
   std::string  aNamePck = mAppli.ICNM()->StdCorrect2
                           (
                               aVH.NameOrKeyHomologues(),
                               mMaster.NameCorrige(),
                               mNameCorrige,
                               true
                           );


    ElPackHomologue aPackV = ElPackHomologue::FromFile(mDir+aNamePck);
    for 
    (
        ElPackHomologue::const_iterator itP = aPackV.begin();
        itP != aPackV.end();
	itP++
    )
    {
        Pt2dr aV12 =  itP->P2() - FromFusionSpace(mMaster.ToFusionSpace(itP->P1()));

	Pt2dr aDifVXY(0,0);
	if  (aVXY)
	{
	    Pt2dr aR2(aTX.getprojR(itP->P1()),aTY.getprojR(itP->P1()));
	    aDifVXY = itP->P2()-aR2;

	    aSDV += euclid(aDifVXY);

	}

        double aDist = euclid(aV12) ;

	aS1++;
	aSD += aDist;
	if (aW)
	{
	   Pt2dr aPC = itP->P1() * aDZ;
	   // aW->draw_circle_abs(aPC,2.0,aW->pdisc()(P8COL::green));
	   Pt2dr aLarg(1,1);
	   aW->fill_rect(aPC-aLarg,aPC+aLarg,aW->pdisc()(P8COL::green));
	   aW->draw_seg(aPC,aPC+aV12*anExag,aW->pdisc()(P8COL::red));
	   if (aVXY)
	       aW->draw_seg(aPC,aPC+aDifVXY*anExagXY,aW->pdisc()(P8COL::blue));
	}
    }

    std::cout << "Dist Moyenne " << aSD / aS1 << "\n";
    std::cout << "Verif Dist Moyenne " << aSDV / aS1 << "\n";

    if (aBW)
    {
       aBW->make_gif(aNameFile.c_str());
       std::cout << "Done bitm "<< aNameFile << "\n";
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
