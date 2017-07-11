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

#include "StdAfx.h"
#include <algorithm>



class cFilterImPolI;
class cResFilterPolI;
class cArgFilterPolI;

//*********************************************

class cArgFilterPolI
{
    public :
      cArgFilterPolI(const std::string & aSymb) :
          mNameIn  (aSymb),
          mBox     (0)
      {
      }

      std::vector<Fonc_Num>      mVIn;
      const std::string          mNameIn;
      std::vector<std::string>   mVArgs;
      Box2di *                   mBox;
};

typedef Fonc_Num  (* tPtrCalcFF)( cFilterImPolI &, const cArgFilterPolI &);

class cResFilterPolI
{
     public :
        
        cResFilterPolI(Fonc_Num aF,Box2di * aBox,std::string aSymbSpec="") :
            mFonc     (aF),
            mBox      (aBox),
            mSymbSpec (aSymbSpec)
        {
        }

        Fonc_Num     mFonc;
        Box2di*      mBox;
        std::string  mSymbSpec;
};


Box2di * UnionBoxPtrWithDel(Box2di * aBox1,Box2di * aBox2)
{
   if (aBox1==0) return aBox2;
   if (aBox2==0) return aBox1;

   Box2di * aRes = new Box2di(Sup(*aBox1,*aBox2));
   delete aBox1;
   delete aBox2;
   return aRes;
}


class cFilterImPolI
{
    public :

       cFilterImPolI(tPtrCalcFF,int aNbFoncIn,int aNbFoncMax,int aNbArgNum,int aNbArgMax,const std::string & aPat);


       tPtrCalcFF mCalc;
       int        mNbFoncIn;
       int        mNbFoncMax;
       int        mNbArgNum;
       int        mNbArgMax;
       std::string mPat;
       cElRegex    mAutom;
};

cFilterImPolI::cFilterImPolI(tPtrCalcFF aCalc,int aNbFoncIn,int  aNbFoncMax,int aNbArgNum,int  aNbArgMax,const std::string & aPat) :
    mCalc      (aCalc),
    mNbFoncIn  (aNbFoncIn),
    mNbFoncMax (aNbFoncMax),
    mNbArgNum  (aNbArgNum),
    mNbArgMax  (aNbArgMax),
    mPat       (aPat),
    mAutom     (mPat,10)
{
}

static double ToDouble(const std::string & aStr)
{
    double aRes;
    FromString(aRes,aStr);
    return aRes;
}

static int ToInt(const std::string & aStr)
{
    int aRes;
    FromString(aRes,aStr);
    return aRes;
}



// Fonc_Num  (* tPtrCalcFF)(cFilterImPolI &,Fonc_Num aFoncIn,const std::string aNameIn,const std::vector<std::string> & aVArgs);


  //----------------------------------------------------------------

static Fonc_Num FAssoc(cFilterImPolI & aFIPI,const cArgFilterPolI & anArg) 
{
  const OperAssocMixte & anOp =   *(OperAssocMixte::GetFromName(anArg.mNameIn));

   Fonc_Num aRes = anOp.opf(anArg.mVIn.at(0),anArg.mVIn.at(1));
   for (int aK=2 ; aK<int(anArg.mVIn.size()) ; aK++)
       aRes = anOp.opf(aRes,anArg.mVIn.at(aK));
   return aRes;
}

static cFilterImPolI  OperAssoc(FAssoc,2,10000,0,0,"\\*|\\+|max|min");

  //----------------------------------------------------------------

static Fonc_Num FOperBin(cFilterImPolI & aFIPI,const cArgFilterPolI & anArg) 
{
    tOperFuncBin  anOper = OperFuncBinaireFromName(anArg.mNameIn);

    return anOper(anArg.mVIn.at(0),anArg.mVIn.at(1));
}

static std::string TheStrOpB="\\-|/|pow|>=|>|<|<=|==|!=|&|&&|(\\|)|(\\|\\|)|\\^|%|mod|>>|<<";
static cFilterImPolI  OperBin(FOperBin,2,2,0,0,TheStrOpB);
  //----------------------------------------------------------------

static Fonc_Num FOperUn(cFilterImPolI & aFIPI,const cArgFilterPolI & anArg) 
{
    tOperFuncUnaire anOper = OperFuncUnaireFromName(anArg.mNameIn);

    return anOper(anArg.mVIn.at(0));
}

static std::string TheStrOpU="--|~|!|signed_frac|ecart_frac|cos|sin|tan|log|log2|exp|square|cube|abs|atan|sqrt|erfcc";
static cFilterImPolI  OperUn(FOperUn,1,1,0,0,TheStrOpU);


  //----------------------------------------------------------------
static Fonc_Num FTif(cFilterImPolI &,const cArgFilterPolI & anArg)
{
   return  Tiff_Im::StdConvGen(anArg.mNameIn,-1,true).in_proj();
}
static cFilterImPolI  OperTif(FTif,0,0,0,0,".*\\.(tif|tiff|jpg|jpeg)");

  //----------------------------------------------------------------

static Fonc_Num FCoord(cFilterImPolI &,const cArgFilterPolI & anArg)
{
   int aKC =0 ;
   if (anArg.mNameIn.size() == 1)
      aKC =  anArg.mNameIn[0]- 'X';
   else
      aKC=  ToInt(anArg.mNameIn.substr(1,std::string::npos));

   return  kth_coord(aKC);
}

static cFilterImPolI  OperCoord(FCoord,0,0,0,0,"X|Y|Z|X[0-9]+");
  //----------------------------------------------------------------

static Fonc_Num FDoubleCste(cFilterImPolI &,const cArgFilterPolI & anArg)
{
   return   Fonc_Num(ToDouble(anArg.mNameIn));
}
static cFilterImPolI  OperDoubleCste(FDoubleCste,0,0,0,0,"-?[0-9]+\\.([0-9]*)?");

static Fonc_Num FIntCste(cFilterImPolI &,const cArgFilterPolI & anArg)
{
   return   Fonc_Num(ToInt(anArg.mNameIn));
}
static cFilterImPolI  OperIntCste(FIntCste,0,0,0,0,"-?[0-9]+");

  //----------------------------------------------------------------

static Fonc_Num FDeriche(cFilterImPolI &,const cArgFilterPolI & anArg)
{
   return   deriche(anArg.mVIn.at(0) ,ToDouble(anArg.mVArgs.at(0)),20);
}

static cFilterImPolI  OperDeriche(FDeriche,1,1,1,1,"deriche");

  //----------------------------------------------------------------

static Fonc_Num FPolar(cFilterImPolI &,const cArgFilterPolI & anArg)
{
   return   polar(anArg.mVIn.at(0),0);
}

static cFilterImPolI  OperPolar(FPolar,1,1,0,0,"polar");

  //----------------------------------------------------------------

static Fonc_Num FExtinc(cFilterImPolI &,const cArgFilterPolI & anArg)
{
    int aD = (anArg.mVArgs.size() >=2) ? ToInt(anArg.mVArgs.at(1)) : 256;
    const Chamfer &  aChmf=  Chamfer::ChamferFromName(anArg.mVArgs.at(0));

    return extinc(anArg.mVIn.at(0),aChmf,aD);
}

static cFilterImPolI  OperExtinc(FExtinc,1,1,1,2,"extinc");


  //----------------------------------------------------------------



static std::vector<cFilterImPolI *>  VPolI()
{
    static std::vector<cFilterImPolI *> aRes;

    if (aRes.size()==0)
    {
         aRes.push_back(&OperBin);
         aRes.push_back(&OperUn);
         aRes.push_back(&OperTif);
         aRes.push_back(&OperAssoc);
         aRes.push_back(&OperCoord);
         // aRes.push_back(&OperPlus);
         // aRes.push_back(&OperMul);
         aRes.push_back(&OperDeriche);
         aRes.push_back(&OperIntCste);
         aRes.push_back(&OperDoubleCste);
         aRes.push_back(&OperPolar);
         aRes.push_back(&OperExtinc);
    }

    return aRes;
}

//=============================

typedef const char * tCPtr;

static int aCptStr=0;
static int aCptCar=0;
static std::string aLast;
static std::string aStrGlob;

static const std::string TheCharSpec="()";

static bool IsCharSpec(const char aCar)
{
  return TheCharSpec.find(aCar)!=std::string::npos;
}

std::string GetString(tCPtr & aStr,bool OnlySpec=false)
{
   std::string aRes;
   //  std::cout << "STRIN=["<< aStr <<"]\n";
   aCptStr++;

   while (*aStr && isspace(*aStr))
   {
       aStr++;
       aCptCar++;
   }

   if (IsCharSpec(*aStr))
   {
       aRes.push_back(aStr[0]);
       aStr++;
       aCptCar++;
       return aRes;
   }
   if (OnlySpec)
      return "";


   while (*aStr && (!isspace(*aStr)) && (! IsCharSpec(*aStr)))
   {
        aRes += *aStr;
        aStr++;
        aCptCar++;
   }

   //  std::cout << "STROUT=["<< aStr <<"]\n";
   //  std::cout << "RES=["<< aRes <<"]\n";
   if (aRes=="")
   {
       std::cout << "After reading [" << aStrGlob.substr(0,aCptCar) << "]\n";
       ELISE_ASSERT(false,"unexcpted end of string");
   }


   aLast = aRes;
   return aRes;
}

cResFilterPolI RecParseStrFNPolI(tCPtr & aStr)
{
    std::string  aSymb = GetString(aStr);

    if (aSymb==")")
    {
        return cResFilterPolI(0,0,")");
    }
/*
*/

    bool aParOuv=false;

    if (aSymb=="(")
    {
       aParOuv=true;
       aSymb = GetString(aStr);
       ELISE_ASSERT((aSymb!="(") && (aSymb!=")") ,"Conesecutive ( in expr");
    }

/*
*/

    std::vector<cFilterImPolI *>  aVPol =  VPolI();

    for (int aK=0 ; aK<int(aVPol.size()) ; aK++)
    {
        cFilterImPolI & aPolI = *(aVPol[aK]);
        if (aPolI.mAutom.Match(aSymb))
        {
            int aNbFonc =  aParOuv ? aPolI.mNbFoncMax : aPolI.mNbFoncIn ;

            cArgFilterPolI anArg(aSymb);
            bool GotParFerm=false;
            for (int aK=0 ; (aK<aNbFonc) && (!GotParFerm)  ; aK++)
            {
                 cResFilterPolI aRFPI = RecParseStrFNPolI(aStr);
                 if (aRFPI.mSymbSpec==")")
                 {
                     GotParFerm =true;
                 }
                 else
                 {
                    anArg.mVIn.push_back(aRFPI.mFonc);
                    anArg.mBox = UnionBoxPtrWithDel(anArg.mBox,aRFPI.mBox);
                 }
            }
            ELISE_ASSERT(int(anArg.mVIn.size())>= aPolI.mNbFoncIn ,"Insufficient number func ");
            ELISE_ASSERT(int(anArg.mVIn.size())<= aPolI.mNbFoncMax ,"Too much number func ");

            int aNbArg =  aParOuv ? aPolI.mNbArgMax : aPolI.mNbArgNum ;
            for (int aK=0 ; aK<aNbArg && (!GotParFerm) ; aK++)
            {
                 std::string aVal = GetString(aStr);
                 if (aVal==")")
                 {
                    GotParFerm =true;
                 }
                 else
                 {
                    anArg.mVArgs.push_back(aVal);
                 }
            }

            if (aParOuv && (! GotParFerm))
            {
                 std::string aVal = GetString(aStr,true);
                 if (aVal==")")
                 {
                     GotParFerm = true;
                 }
            }

            if (GotParFerm!=aParOuv)
            {
               std::cout << aParOuv << " " << GotParFerm << "\n";
               ELISE_ASSERT(false,"Bad parenthese match");
            }
            ELISE_ASSERT(int(anArg.mVArgs.size())>= aPolI.mNbArgNum ,"Insufficient number arg");
            ELISE_ASSERT(int(anArg.mVArgs.size())<= aPolI.mNbArgMax ,"Too much number arg");

            if (&aPolI==&OperTif)
            {
               Tiff_Im aTif =  Tiff_Im::StdConvGen(aSymb,-1,true);
               anArg.mBox = new Box2di(Pt2di(0,0),aTif.sz());
            }

            return  cResFilterPolI(aPolI.mCalc(aPolI,anArg),anArg.mBox);
        }
    }

    std::cout << "For symb=[" << aSymb << "]\n";
    ELISE_ASSERT(false,"could not recognize symbol");

    return * ((cResFilterPolI *) 0);
}

cResFilterPolI GlobParseStrFNPolI(tCPtr & aStr)
{
    aStrGlob = aStr;

    return RecParseStrFNPolI(aStr);
}


int Nikrup_main(int argc,char ** argv)
{
    std::string anExpr;
    std::string aNameOut;
    Box2di       aBoxOut;
    // GenIm::type_el aType=  GenIm::real4;
    std::string aNameTypeOut = "real4";
    int aNbChan;

    ElInitArgMain
    (
         argc,argv,
         LArgMain()  << EAMC(anExpr,"Expression")
                     << EAMC(aNameOut,"File for resulta"),
         LArgMain()  << EAM(aBoxOut,"Box",true,"Box of result, def computed according to files definition")
                     << EAM(aNbChan,"NbChan",true,"Number of output chan, def 3 if input %3, else 1")
                     << EAM(aNameTypeOut,"Type",true,"Type of output, def=real4 ")
    );

    GenIm::type_el aType=  type_im(aNameTypeOut);
    // std::cout << "EXPR=[" << anExpr << "]\n";

    // anExpr = anExpr + " ";
    
     const char * aCPtr = anExpr.c_str();
     cResFilterPolI aRFPI =  GlobParseStrFNPolI(aCPtr);

     if (!EAMIsInit(&aBoxOut))
     {
         ELISE_ASSERT(aRFPI.mBox,"Cannot compute size");
         aBoxOut = *aRFPI.mBox;
     }
     Pt2di aSzOut = aBoxOut.sz();
     Fonc_Num aFonc = aRFPI.mFonc;

     int aDimOut = aFonc.dimf_out();
     if (! EAMIsInit(&aNbChan))
     {
         aNbChan = (aDimOut%3) ? 1 : 3;
     }

     ELISE_ASSERT((aNbChan==1) || (aNbChan==3),"Cant handle 1 or 3 channel as out");
     ELISE_ASSERT((aDimOut%aNbChan)==0,"Nb channel sepcified is not a divisor of channel got");
      

     int aNbOut = aDimOut / aNbChan;


     Tiff_Im::PH_INTER_TYPE aPIT = (aNbChan==1) ? Tiff_Im::BlackIsZero : Tiff_Im::RGB;

     Output  anOutGlob = Output::onul(0);

     for (int aK=0 ; aK<aNbOut ; aK++)
     {
          std::string aNameK =  aNameOut;
          if (aNbOut>1)
             aNameK = "Nkrp-" + ToString(aK) + aNameOut;

          Tiff_Im aTifOut
                 (
                    aNameK.c_str(),
                    aSzOut,
                    aType,
                    Tiff_Im::No_Compr,
                    aPIT
                 );
          Output anOutK = aTifOut.out();

          anOutGlob = (aK==0) ?  anOutK : Virgule(anOutGlob,anOutK);
     }


     ELISE_COPY
     (
         rectangle(Pt2di(0,0),aSzOut),
         trans(aFonc,aBoxOut._p0),
         anOutGlob
     );

     return EXIT_SUCCESS;
}

#if (0)
#endif

// static cFilterImPolI  TheFilterPlus(,





#define DEF_OFSET -12349876


int Contrast_main(int argc,char ** argv)
{
    std::string aNameIn,aNameOut,aNameMasq;
    std::vector<double> aVSzW;
    std::vector<int>    aVNbIter;
    std::vector<double> aVPds;
    bool Gray = true;


    ElInitArgMain
    (
         argc,argv,
         LArgMain()  << EAMC(aNameIn,"Name of Input image", eSAM_IsExistFile)
                     << EAMC(aVSzW,"Sizes of window")
                     << EAMC(aVNbIter,"Number of iterations")
                     << EAMC(aVPds,"Weighting"),
         LArgMain()  << EAM(aNameOut,"Out",true,"Name of Result", eSAM_NoInit)
                     << EAM(Gray,"Gray",true,"Gray image , def=true", eSAM_IsBool)
                     << EAM(aNameMasq,"Masq",true,"Masq of image", eSAM_IsBool)
    );

    if (!MMVisualMode)
    {
        ELISE_ASSERT(aVSzW.size()==aVNbIter.size(),"Size vect incoherent");
        ELISE_ASSERT(aVSzW.size()==aVPds.size(),"Size vect incoherent");
        if (! EAMIsInit(&aNameOut) ) 
           aNameOut = "FiltreContrast-" + aNameIn;

        Tiff_Im aTiffIn = Tiff_Im::StdConvGen(aNameIn,Gray ? 1 : 3,true);

        Fonc_Num aFoncMasq = aTiffIn.inside();

        if (EAMIsInit(&aNameMasq))
           aFoncMasq = aFoncMasq * Tiff_Im(aNameMasq.c_str()).in(0);

        Fonc_Num aF = aTiffIn.in(0) * aFoncMasq;


         Fonc_Num aFAd = 0.0;
         Fonc_Num aSomP = 0.0;

          

         for (int aK=0 ; aK <int(aVSzW.size()) ; aK++)
         {
              int aSzW = aVSzW[aK];
              int aNbIter = aVNbIter[aK];
              double aPds = aVPds[aK];
              Fonc_Num aFiltre = Rconv(aF);
              for (int aK=0 ; aK<aNbIter  ; aK++)
                  aFiltre = rect_som(aF*aFoncMasq,aSzW) / Max(1e-5, rect_som(aFoncMasq,aSzW));

               aF =  aF * (aPds+1) - aFiltre * aPds;
         }

         Tiff_Im::CreateFromFonc
         (
             aNameOut,
             aTiffIn.sz(),
             aF,
             GenIm::real4
         );
        

/*
        Tiff_Im aFileIm = Tiff_Im::UnivConvStd(aNameIn.c_str());
        Pt2di aSzIm = aFileIm.sz();
        int aNBC = aFileIm.nb_chan();

        std::vector<Im2D_REAL4> aVIm;
        Output anOut = Output::onul(0);
        for (int aK=0 ; aK<aNBC ; aK++)
        {
            Im2D_REAL4  anIm(aSzIm.x,aSzIm.y);
            aVIm.push_back(anIm);
            anOut = (aK==0) ? anIm.out() : Virgule(anOut,anIm.out());
        }
        ELISE_COPY(aFileIm.all_pts(),aFileIm.in(),anOut);


        Tiff_Im aFileMasq(aNameMasqOK.c_str());
        Im2D_Bits<1> aMasq(aSzIm.x,aSzIm.y,1);
        ELISE_COPY(aFileMasq.all_pts(),!aFileMasq.in_bool(),aMasq.out());


        Im2D_Bits<1> aMasq2Fill(aSzIm.x,aSzIm.y,1);
        if (EAMIsInit(&aNameMasq2FIll))
        {
            Tiff_Im aFileMasq(aNameMasq2FIll.c_str());
            ELISE_COPY(aFileMasq.all_pts(),!aFileMasq.in_bool(),aMasq2Fill.out());
        }


        Fonc_Num aFRes=0;
        for (int aK=0 ; aK<aNBC ; aK++)
        {
            aVIm[aK] = ImpaintL2(aMasq,aMasq2Fill,aVIm[aK]);
            aFRes = (aK==0) ? aVIm[aK].in() : Virgule(aFRes,aVIm[aK].in());
        }

        if (!EAMIsInit(&aNameOut))
        {
            aNameOut = StdPrefix(aNameIn) + "_Impaint.tif";
        }
        Tiff_Im aTifOut
                (
                    aNameOut.c_str(),
                    aSzIm,
                    aFileIm.type_el(),
                    Tiff_Im::No_Compr,
                    aFileIm.phot_interp()
                    );

        ELISE_COPY(aTifOut.all_pts(),aFRes,aTifOut.out());
*/
    }
    return EXIT_SUCCESS;
}





/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
