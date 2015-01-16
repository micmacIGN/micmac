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
#include "Traj_Aj.h"

using namespace NS_AJ;

/**************************************************************/
/*                                                            */
/*                 cTAj2_OneLogIm                             */
/*                                                            */
/**************************************************************/

int GetExprDef(cElRegex & anAutom,const cTplValGesInit<int> & aK,const cTplValGesInit<int>  & aDef)
{
   if (aK.IsInit())
      return anAutom.VNumKIemeExprPar(aK.Val());
   return aDef.Val();
}

const ElMatrix<double> & cTAj2_OneLogIm::MatI2C() const
{
   return mMatI2C;
}

cTAj2_OneLogIm::cTAj2_OneLogIm
(
    cAppli_Traj_AJ & anAppli,
    int                 aKLine,
    cTAj2_LayerLogIm &  aLayer,
    const cTrAJ2_SectionLog & aParam,
    const std::string & aLine
) :
   mAppli      (anAppli),
   mKLine      (aKLine),
   mLayer      (aLayer),
   mParam      (aParam),
   mLine       (aLine),
   mTimeIsInit (false),
   mKeyIm      (""),
   mKeyImIsInit(false),
   mHasTeta    (false),
   mMatI2C     (ElMatrix<double>::Rotation(0,0,0))
{
    // std::cout << mKLine << "L=" << mLine << "\n";
    if (!  aLayer.Autom().Match(aLine))
    {
        std::cout << "Line=" << mLine << "\n";
        std::cout << "Autom=" << aParam.Autom() << "\n";
        ELISE_ASSERT(false,"No Match");
    }
    else
    {
    }

    const cSectionTime & aST = aParam.SectionTime();
    if (aST.KTime().IsInit())
    {
       mTimeIsInit = true;
       mTime =  aLayer.Autom().VNumKIemeExprPar(aST.KTime().Val());
    }
    else if (aST.FullDate().IsInit())
    {
        mTimeIsInit = true;
        static cElDate aT0(1,1,2011,cElHour(0,0,0));

        const cFullDate & aFD = aST.FullDate().Val();

        int aYear =  GetExprDef(aLayer.Autom(),aFD.KYear(),aFD.DefYear());
        int aMonth =  GetExprDef(aLayer.Autom(),aFD.KMonth(),aFD.DefMonth());
        int aDay =  GetExprDef(aLayer.Autom(),aFD.KDay(),aFD.DefDay());

        int aHour =  aLayer.Autom().VNumKIemeExprPar(aFD.KHour());
        int aMin  =  aLayer.Autom().VNumKIemeExprPar(aFD.KMin());
        double aSec = aLayer.Autom().VNumKIemeExprPar(aFD.KSec()) / aFD.DivSec().Val();
        if (aFD.KMiliSec().IsInit())
            aSec += aLayer.Autom().VNumKIemeExprPar(aFD.KMiliSec().Val()) / (1000.0 * aFD.DivMiliSec().Val());

       cElDate  aT(aDay,aMonth,aYear,cElHour(aHour,aMin,aSec));
       mTime = aT.DifInSec(aT0);


        // int aD = 
        // cElDate aD( aLayer.Autom().VNumKIemeExprPar(
    }
    else if (aST.NoTime().IsInit())
    {
    }



    mT0 = mTime;


    std::vector<double> aVC;
    aVC.push_back(aLayer.Autom().VNumKIemeExprPar(aParam.KCoord1())/aParam.DivCoord1().Val());
    aVC.push_back(aLayer.Autom().VNumKIemeExprPar(aParam.KCoord2())/aParam.DivCoord2().Val());
    aVC.push_back(aLayer.Autom().VNumKIemeExprPar(aParam.KCoord3())/aParam.DivCoord3().Val());


    aVC = VecCorrecUnites(aVC,aParam.UnitesCoord());
    for (int aK=0 ; aK<3 ; aK++)
    {
        mCoord[aK] =  aVC[aK];
    }

    if (aParam.TrajAngles().IsInit())
    {
        mHasTeta = true;
        const cTrajAngles & aTrA = aParam.TrajAngles().Val();
        if (aTrA.RefOrTrajI2C().IsInit())
        {
              mMatI2C = Std_RAff_C2M(aTrA.RefOrTrajI2C().Val(),eConvOriLib);
        }
        eUniteAngulaire aUnit = aTrA.Unites();
        int aKT1 = aTrA.KTeta1();
        mTetas[0] = (aKT1<0) ? 0.0 : ToRadian(aLayer.Autom().VNumKIemeExprPar(aKT1),aUnit) +aTrA.OffsetTeta1().Val();
        int aKT2 = aTrA.KTeta2();
        mTetas[1] =  (aKT2<0) ? 0.0 :  ToRadian(aLayer.Autom().VNumKIemeExprPar(aKT2),aUnit) +aTrA.OffsetTeta2().Val();
        int aKT3 = aTrA.KTeta3();
        mTetas[2] =   (aKT3<0) ? 0.0 : ToRadian(aLayer.Autom().VNumKIemeExprPar(aKT3),aUnit) +aTrA.OffsetTeta3().Val();

        if (aTrA.ConvOr() == eConvAngAvionJaune) 
        {
           // Cap au nord, anti trigo ...
               mTetas[0] = PI/2.0 - mTetas[0];
               mTetas[0] = angle_mod_real(mTetas[0],2*PI);
        }
        else if (aTrA.ConvOr() == eConvAngSurvey)
        {
           // Cap au nord, anti trigo ...
                mTetas[0] = -PI/2.0 - mTetas[0];
/*
               mTetas[0] =   - mTetas[0];
               mTetas[0] = angle_mod_real(mTetas[0],2*PI);
std::cout <<  " SURVEY ANGLE NEW  \n";
*/
        }
        else if (aTrA.ConvOr() == eConvApero_DistC2M) 
        {
        }
/*
        else if (aTrA.ConvOr() == eConvAngErdas) 
        {
            mTetas[1] = - mTetas[1] ; //  std::cout << "WWWWWWWWWWWWww----------------\n";
        }
*/
        else
        {
            std::cout << "For Conv " << eToString(aTrA.ConvOr()) << "\n";
            ELISE_ASSERT(false,"Un suported conv");
        }
 
        {
           double aF = 180/PI;
           if (mAppli.TraceLog(*this))
           {
              printf("%d Tetas : %lf %lf %lf \n",mKLine,mTetas[0]*aF,mTetas[1]*aF,mTetas[2]*aF);
           }
        }
    }

    if ( aParam.GetImInLog().IsInit())
    {
        const cGetImInLog & aGIIL = aParam.GetImInLog().Val();
        mKeyImIsInit = true;
        mKeyIm =  aLayer.Autom().KIemeExprPar(aGIIL.KIm());
    }
    // std::cout << mTime <<  " " << mTetas[0] << "\n";
}

const std::string & cTAj2_OneLogIm::KeyIm() const
{
   ELISE_ASSERT(mKeyImIsInit,"cTAj2_OneLogIm::KeyIm No Init");
   return mKeyIm;
}

double cTAj2_OneLogIm::Time() const
{
   ELISE_ASSERT(mTimeIsInit,"cTAj2_OneLogIm::Time not init");
   return mTime;
}

double cTAj2_OneLogIm::T0() const
{
   return mT0;
}

Pt3dr  cTAj2_OneLogIm::PCBrut() const
{
   return Pt3dr(mCoord[0],mCoord[1],mCoord[2]);
}

Pt3dr  cTAj2_OneLogIm::PGeoC() const
{
   return mLayer.SC()->ToGeoC(PCBrut());
}

void cTAj2_OneLogIm::InitT0(const cTAj2_OneLogIm & aLogBase)
{
    mT0 = mTime - aLogBase.mTime;
    // std::cout << mT0 << "\n";
}

void cTAj2_OneLogIm::ResetMatch()
{
    mLKM.Reset();
}

void cTAj2_OneLogIm::UpdateMatch(cTAj2_OneImage * anIm,double aDif)
{
   mLKM.Update(anIm,aDif);
}

cTAj2_OneImage * cTAj2_OneLogIm::BestMatch()
{
    return mLKM.mBestMatch;
}

eTypeMatch  cTAj2_OneLogIm::QualityMatch(double aDif)
{
   return mLKM.QualityMatch(aDif,this);
}

int  cTAj2_OneLogIm::KLine() const
{
    return mKLine;
}

double cTAj2_OneLogIm::Teta(int aK) const
{
    ELISE_ASSERT(mHasTeta&&(aK>=0)&&(aK<3),"cTAj2_OneLogIm::Teta");
    return mTetas[aK];
}

class cCmpLogTime
{
     public :
       bool operator()(const cTAj2_OneLogIm *  aI1,const cTAj2_OneLogIm *  aI2)
       {
             return aI1->Time() < aI2->Time();
       }
};

/**************************************************************/
/*                                                            */
/*               cTAj2_LayerLogIm                             */
/*                                                            */
/**************************************************************/

#define TBUF 1000

#define BCP 1e30
#define PEU -1e30

cTAj2_LayerLogIm::cTAj2_LayerLogIm
(
    cAppli_Traj_AJ & anAppli,
    const cTrAJ2_SectionLog&  aSL
) :
   mAppli (anAppli),
   mSL    (aSL),
   mRegEx (new cElRegex(aSL.Autom(),30)),
   mSC    (cSysCoord::FromXML(aSL.SysCoord(),mAppli.DC().c_str())),
   mCoordMin  (BCP,BCP,BCP),
   mCoordMax  (PEU ,PEU,PEU)
{
    std::string aNF = mAppli.DC() + aSL.File();
    ELISE_fp  aFP (aNF.c_str(),ELISE_fp::READ);

    char aBUF[TBUF];
    bool GotEOF;
    int aKLine=0;
    while (aFP.fgets(aBUF,TBUF,GotEOF) && (!GotEOF))
    {
        for (char * aC=aBUF; *aC ; aC++)
        {
            if (isspace(*aC)) *aC = ' ' ;
        }
        cTAj2_OneLogIm * aLog= new cTAj2_OneLogIm(mAppli,aKLine,*this,aSL,std::string(aBUF));
        if (aSL.TimeMin().IsInit() && (aLog->Time()  < aSL.TimeMin().Val())   )
        {
           delete aLog;
        }
        else
        {
           mLogs.push_back(aLog);
           mCoordMin = Inf(mCoordMin,aLog->PCBrut());
           mCoordMax = Sup(mCoordMax,aLog->PCBrut());
        }
        aKLine++;
    }
    aFP.close();


    if ( ! aSL.SectionTime().NoTime().IsInit())
    {
        cCmpLogTime aCmpTimeLog;
        std::sort(mLogs.begin(),mLogs.end(),aCmpTimeLog);

        cTAj2_OneLogIm * aLogT0 = KthLog(aSL.KLogT0().Val());
        for (int aK=0 ; aK<int(mLogs.size()) ; aK++)
        {
           mLogs[aK]->InitT0(*aLogT0);
        }
    }

    if (0)
    {
       for (int aK=1 ; aK<int(mLogs.size()) ; aK++)
       {
           cTAj2_OneLogIm * aP0 = mLogs[aK-1];
           cTAj2_OneLogIm * aP1 = mLogs[aK];
           std::cout << "DT " <<  (aP1->Time()-aP0->Time()) << " Dist " << euclid(aP1->PGeoC()-aP0->PGeoC()) << "\n";
       }
    }


    for 
    (
         std::list<cGenerateTabExemple>::const_iterator itG=aSL.GenerateTabExemple().begin();
         itG!=aSL.GenerateTabExemple().end();
         itG++
    )
    {
          GenerateOneExample(*itG);
    }
}

double  GeneratePdsBar(int aNb,int aK,bool aRand)
{
    return aRand ? NRrandom3() :  double(aK) / double (aNb);
}

void cTAj2_LayerLogIm::GenerateOneExample(const cGenerateTabExemple & aGTE)
{
    std::string aName = mAppli.DC()+aGTE.Name();
    FILE * aFP = FopenNN(aName.c_str(),"w","cTAj2_LayerLogIm::GenerateOneExample");
    Pt3dr aP0 = mCoordMin;
    Pt3dr aP1 = mCoordMax;
    aP0.z = aGTE.ZMin().ValWithDef(aP0.z-aGTE.DIntervZ().Val());
    aP1.z = aGTE.ZMax().ValWithDef(aP1.z+aGTE.DIntervZ().Val());

    int aCpt=0;
    for (int aKx=0 ; aKx <=aGTE.Nb().x ; aKx++)
    {
       for (int aKy=0 ; aKy <=aGTE.Nb().y ; aKy++)
       {
          for (int aKz=0 ; aKz <=aGTE.Nb().z ; aKz++)
          {
                double aPdx = GeneratePdsBar(aGTE.Nb().x,aKx,aGTE.RandomXY().Val());
                double aPdy = GeneratePdsBar(aGTE.Nb().y,aKy,aGTE.RandomXY().Val());
                double aPdz = GeneratePdsBar(aGTE.Nb().z,aKz,aGTE.RandomZ().Val());

                Pt3dr aQ (
                              aP0.x*aPdx + aP1.x*(1-aPdx),
                              aP0.y*aPdy + aP1.y*(1-aPdy),
                              aP0.z*aPdz + aP1.z*(1-aPdz)
                         );
                fprintf(aFP,"%d %f %f %f\n",aCpt,aQ.x,aQ.y,aQ.z);
                aCpt++;
          
          }
       }
    }
    ElFclose(aFP);
}


void cTAj2_LayerLogIm::ResetMatch()
{
   for (int aK=0; aK<int(mLogs.size()) ; aK++)
   {
      mLogs[aK]->ResetMatch();
   }
}

int  cTAj2_LayerLogIm::NbLog() const
{
   return mLogs.size();
}

cTAj2_OneLogIm * cTAj2_LayerLogIm::KthLog(int aK) const
{
   return mLogs.at(aK);
}


cElRegex & cTAj2_LayerLogIm::Autom()
{
  return *mRegEx;
}

const cSysCoord *  cTAj2_LayerLogIm::SC()
{
   return mSC;
}


std::vector<cTAj2_OneLogIm *> & cTAj2_LayerLogIm::Logs()
{
   return mLogs;
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
