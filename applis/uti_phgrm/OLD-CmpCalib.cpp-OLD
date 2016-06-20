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
#include "XML_GEN/all.h"
#include <algorithm>

using namespace NS_ParamChantierPhotogram;


#define DEF_OFSET -12349876

class cCapteurCmpCal
{
    public :
        cCapteurCmpCal(const std::string & aName) :
             mName (aName)
	{
	}

	static cCapteurCmpCal & StdAlloc(const std::string & aName);

        const std::string & Name() { return mName; }
        virtual Pt2dr P0()                               = 0;
        virtual Pt2dr P1()                               = 0;
	virtual double  Focale()                         = 0;
        virtual Pt2dr ToPDirL3(const Pt2dr&  aP) const   = 0;
    private :
        std::string   mName;
};


class cGridCmpCal : public cCapteurCmpCal
{
    public :
        cGridCmpCal(const std::string & aName)  :
             cCapteurCmpCal(aName)
        {
             SplitDirAndFile(mDir,mFile,aName);
             std::cout << "[" << mDir << "][" << mFile << "]\n";
             cDbleGrid::cXMLMode aXM;
             mGr = new cDbleGrid(aXM,mDir,mFile);
        }

        //  const std::string & Name() { return mName; }
        Pt2dr P0() {return mGr->GrDir().P0();}
        Pt2dr P1() {return mGr->GrDir().P1();}
	double  Focale() {return mGr->Focale();}
        Pt2dr ToPDirL3(const Pt2dr&  aP) const {return mGr->Direct(aP);}


   private :
        std::string   mDir;
        std::string   mFile;
        cDbleGrid   * mGr;
};

class cCamStenopeCmpCal : public cCapteurCmpCal
{
    public :
        cCamStenopeCmpCal(const std::string & aName)  :
             cCapteurCmpCal(aName),
	     mCam (Std_Cal_From_File(aName))
        {
	    std::cout << aName << "\n";
        }


        //  const std::string & Name() { return mName; }
        Pt2dr P0() {return Pt2dr(0,0);}
        Pt2dr P1() {return Pt2dr(mCam->Sz());}
	double  Focale() {return  mCam->Focale();}
        Pt2dr ToPDirL3(const Pt2dr&  aP) const {return mCam->F2toPtDirRayonL3(aP);}


   private :
       CamStenope * mCam;
};

cCapteurCmpCal & cCapteurCmpCal::StdAlloc(const std::string & aName)
{
   cElXMLTree aTr(aName);

   if (aTr.Get("sensor"))
       return  * (new cGridCmpCal(aName));

   return * (new cCamStenopeCmpCal(aName));
}

class cAppliCmpCal
{
     public :
          cAppliCmpCal
          (
                 const std::string & aName1,
                 const std::string & aName2,
                 bool  aModeL1,
                 int   aSzW,
                 double aDynV
          ) :
             mGr1   (cCapteurCmpCal::StdAlloc(aName1)),
             mGr2   (cCapteurCmpCal::StdAlloc(aName2)),
             mSetEq   (aModeL1 ? cNameSpaceEqF::eSysL1Barrodale : cNameSpaceEqF::eSysPlein,1000),
             mEqORV (mSetEq.NewEqObsRotVect()),
             mRotF  (mEqORV->RotF()),
             mNBP   (10) ,
             mBrd   (0.02),

             mP0    (Inf(mGr1.P0(), mGr2.P0())),
             mP1    (Inf(mGr1.P1(), mGr2.P1())),
             mSz     (mP1-mP0),
             mRatioW (aSzW/ElMax(mSz.x,mSz.y)),

             mBox    (mP0,mP1),
             mMil    (mBox.milieu()),
             mRay    (euclid(mP0,mMil)),
             mFocale ( (mGr1.Focale() + mGr2.Focale()) / 2.0),
             mSzW    (round_ni(mSz*mRatioW)),
             mW      (aSzW>0 ? Video_Win::PtrWStd(mSzW) : 0),
             mDynVisu (aDynV),
             mRotCur  (Pt3dr(0,0,0),0,0,0)
          {
std::cout <<  mGr1.P0() << mGr1.P1() << "\n";
std::cout <<  mGr2.P0() << mGr2.P1() << "\n";
              mSetEq.SetClosed();
          }

          void OneItere(bool First,bool Last);

          double K2Pds(int aK)
          {
              return mBrd +  (1-2*mBrd) * aK/double(mNBP);
          }
          void InitNormales(Pt2dr aPIm);
          Pt2dr EcartNormaleCorr();
          Pt2dr ENC_From_PIm(Pt2dr aPIm);
          double EcartFromRay(double aR);

     private :
          cCapteurCmpCal & mGr1;
          cCapteurCmpCal & mGr2;
          cSetEqFormelles  mSetEq;
          cEqObsRotVect *  mEqORV;
          cRotationFormelle & mRotF;
          int               mNBP;
          double            mBrd;
          Pt2dr             mP0;
          Pt2dr             mP1;
          Pt2dr             mSz;
          double            mRatioW;
          Box2dr            mBox;
          Pt2dr             mMil;
          double            mRay;
          double            mFocale;
          Pt2di             mSzW;
          Video_Win *       mW;
          double            mDynVisu;
          ElRotation3D      mRotCur;
          Pt3dr             mN1;
          Pt3dr             mN2;
};

void cAppliCmpCal::InitNormales(Pt2dr aPIm)
{
    Pt2dr aPPH1 =  mGr1.ToPDirL3(aPIm);
    Pt2dr aPPH2 =  mGr2.ToPDirL3(aPIm);
    mN1 = PZ1(aPPH1);
    mN2 = PZ1(aPPH2);
    mN1 = mN1 / euclid(mN1);
    mN2 = mN2 / euclid(mN2);
}

Pt2dr cAppliCmpCal::EcartNormaleCorr()
{
   return (ProjStenope(mRotCur.ImVect(mN1))-ProjStenope(mN2))*mFocale;
}

Pt2dr cAppliCmpCal::ENC_From_PIm(Pt2dr aPIm)
{
    InitNormales(aPIm);
    return EcartNormaleCorr();
}

double cAppliCmpCal::EcartFromRay(double aR)
{
    int aNbDir = 3000;

    double aS1=0, aSD=0;
    for (int aK=0 ; aK<aNbDir ; aK++)
    {
        Pt2dr aP = mMil + Pt2dr::FromPolar(aR,(2*PI*aK)/aNbDir);
        if (mBox.inside(aP))
        {
            aS1++;
            double aD = euclid(ENC_From_PIm(aP));
            if (aD>1.0)
            {
                 // std::cout << "P= " << aP << ENC_From_PIm(aP)  << "\n";
                 // getchar();
            }
            aSD += aD;
        }
    }
    return (aS1>0) ? (aSD/aS1) : -1;
}


void cAppliCmpCal::OneItere(bool First,bool Last)
{

    double aS1=0, aSD=0;
    mSetEq.AddContrainte(mRotF.StdContraintes(),true);
    mSetEq.SetPhaseEquation();
    mRotCur =  mRotF.CurRot();

    FILE * aFP=0;
    if (Last)
    {
        std::string aName = StdPrefix(mGr1.Name()) + "_Ecarts.txt";
        aFP = ElFopen(aName.c_str(),"w");
    }
 
    if (Last)
    {
       double aStep = 200;
       fprintf(aFP,"--------------  Ecart radiaux  -----------\n");
       fprintf(aFP," Rayon   Ecart\n");
       for( double aR = 0 ; aR<(mRay+aStep) ; aR+= aStep)
       {
           double aRM = ElMin(aR,mRay-10.0);
           double EC =  EcartFromRay(aRM);
           std::cout << "Ray=" << aRM 
                     << " ; Ecart=" << EC << "\n";
           fprintf(aFP," %lf %lf\n",aRM,EC);
       }
    }

    if (Last)
    {
       fprintf(aFP,"--------------  Ecart plani  -----------\n");
       fprintf(aFP,"Im.X Im.Y  PhG.X Phg.Y Ec\n");
    }
    for (int aKX=0 ; aKX<=mNBP ; aKX++)
       for (int aKY=0 ; aKY<=mNBP ; aKY++)
       {
           double aPdsX = K2Pds(aKX);
           double aPdsY = K2Pds(aKY);

           Pt2dr  aPIm (
                      mP0.x * aPdsX+mP1.x * (1-aPdsX),
                      mP0.y * aPdsY+mP1.y * (1-aPdsY)
                 );
 
           InitNormales (aPIm);

           mEqORV->AddObservation(mN1,mN2);
           Pt2dr U = EcartNormaleCorr();
           aS1++;
           aSD += euclid(U);

            if ((mW!=0) && (First || Last ))
            {
                // Pt2dr aP0 (mSzW*aPdsX,mSzW*aPdsY);
               //  Pt2dr aP0 = (aPIm- mP0)  * mRatioW;
                Pt2dr aP0 = (aPIm-mP0)  * mRatioW;

 
                mW->draw_circle_loc(aP0,2.0,mW->pdisc()(P8COL::green));
                int aCoul = First ? P8COL::blue : P8COL::red;
               
                mW->draw_seg
                (
                   aP0,
                   aP0 + U* mDynVisu,
                   mW->pdisc()(aCoul)
                );
            }
            if (aFP) 
               fprintf(aFP,"%lf %lf %lf %lf %lf\n",aPIm.x,aPIm.y,U.x,U.y,euclid(U));
       }

       std::cout << (aSD/aS1) << " "  << (aSD/aS1) * (1e6/mFocale) << " MicroRadians " << "\n";
       mSetEq.SolveResetUpdate();

       if (aFP)
          ElFclose(aFP);
}


int main(int argc,char ** argv)
{


    double  aTeta01 = 0.0;
    double  aTeta02 = 0.0;
    double  aTeta12 = 0.0;
    int     aL1 = 0;
    int     aSzW= 700;
    double  aDynV = 100.0;
    std::string aName1;
    std::string aName2;

    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAM(aName1) 
	            << EAM(aName2) ,
	LArgMain()  << EAM(aTeta01,"Teta01",true)
	            << EAM(aTeta02,"Teta02",true)
	            << EAM(aTeta12,"Teta12",true)
                    << EAM(aL1,"L1",true)
                    << EAM(aSzW,"SzW",true)
                    << EAM(aDynV,"DynV",true)
    );	

    cAppliCmpCal aCmpC(aName1,aName2,aL1,aSzW,aDynV);

    int aNbStep = 5;
    for (int aK=0 ; aK< 5 ; aK++)
        aCmpC.OneItere(aK==0,aK==(aNbStep-1));

  getchar();
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
