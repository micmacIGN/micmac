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
class cCtxtFoncPolI;
typedef const char * tCPtr;

cResFilterPolI RecParseStrFNPolI(tCPtr & aStr,cCtxtFoncPolI * aCtx);

std::string EndStr(const std::string & aStr,int aNb=1)
{
    return aStr.substr(1,std::string::npos);
}

/*
std::string StrToLower(const std::string& aIn)
{
    std::string aRes;

    for (const char * aC=aIn.c_str() ; *aC ; aC++)
       aRes += *aC;

    return aRes;
}
*/


//*********************************************
//*********************************************
//*********************************************

void Nikrup_Banniere()
{
    std::cout <<  "\n";
    std::cout <<  " *********************************\n";
    std::cout <<  " *     N-ew                      *\n";
    std::cout <<  " *     I-mage                    *\n";
    std::cout <<  " *     K-it                      *\n";
    std::cout <<  " *     R-everse                  *\n";
    std::cout <<  " *     U-se of                   *\n";
    std::cout <<  " *     P-olish notation          *\n";
    std::cout <<  " *********************************\n\n";

}

class cCtxtFoncPolI
{
    public :
        Symb_FNum* Add(const std::string& aSymb,Fonc_Num aF,bool Svp);
        cCtxtFoncPolI * Dup();
        cCtxtFoncPolI();
        Symb_FNum* GetValSymb(const std::string& aName);
        void  Herit(cCtxtFoncPolI *);
    private :
        

        std::map<std::string,Fonc_Num>   mMapF;
        std::map<std::string,Symb_FNum*> mMapS;
        cCtxtFoncPolI(const cCtxtFoncPolI&);  // N.I.
};
cCtxtFoncPolI::cCtxtFoncPolI()
{
}

Symb_FNum* cCtxtFoncPolI::GetValSymb(const std::string& aName)
{
    Symb_FNum* aSFN = mMapS[aName];
    if (aSFN==0)
    {
        std::cout << "For name=[" << aName << "]" << this << "\n";
        ELISE_ASSERT(false,"Symbol has no value");
    }
    return aSFN;
}


Symb_FNum* cCtxtFoncPolI::Add(const std::string& aSymb,Fonc_Num aF,bool Svp=false)
{
    if (DicBoolFind(mMapF,aSymb))
    {
        if (Svp) return mMapS[aSymb] ;
        std::cout << "For symb=" << aSymb << "\n";
        ELISE_ASSERT(false,"Multiple symb def");
    }
    mMapF[aSymb] = aF;
    Symb_FNum*  aRes = new Symb_FNum(aF);
    mMapS[aSymb] = aRes;
    return aRes;
}
cCtxtFoncPolI * cCtxtFoncPolI::Dup()
{
   cCtxtFoncPolI * aRes = new cCtxtFoncPolI;
   
   aRes->Herit(this);
/*
   for (std::map<std::string,Fonc_Num>::iterator it=mMapF.begin(); it!=mMapF.end() ; it++)
   {
       aRes->Add(it->first,it->second);
   }
*/
   return aRes;
}

void  cCtxtFoncPolI::Herit(cCtxtFoncPolI * aH)
{
   for (std::map<std::string,Fonc_Num>::iterator it=aH->mMapF.begin(); it!=aH->mMapF.end() ; it++)
   {
       Add(it->first,it->second,true);
   }
}


class cArgFilterPolI
{
    public :
      cArgFilterPolI(const std::string & aSymb,cCtxtFoncPolI * aCtx) :
          mNameIn  (aSymb),
          mBox     (0),
          mCtx     (aCtx)
      {
      }

      std::vector<Fonc_Num>      mVIn;
      const std::string          mNameIn;
      std::vector<std::string>   mVArgs;
      Box2di *                   mBox;
      cCtxtFoncPolI *            mCtx;
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

       // mChgCtx => signifie que les Symbole Elise ne peuvent pas etre reutilise et qu'il faut 
       // duplique la fonction

       cFilterImPolI(tPtrCalcFF,int aNbFoncIn,int aNbFoncMax,int aNbArgNum,int aNbArgMax,const std::string & aPat,bool ChgCtx,const std::string & aCom);


       void Show() const;


       tPtrCalcFF  mCalc;
       int         mNbFoncIn;
       int         mNbFoncMax;
       int         mNbArgNum;
       int         mNbArgMax;
       std::string mPat;
       cElRegex    mAutom;
       bool        mChgCtx;
       std::string mCom;
};

void cFilterImPolI::Show() const
{
           std::cout << "====== Name=[" << mPat << "]" ;
           std::cout << "\n";
           std::cout << " Com=" << mCom  << " ;  NbF=" ;
           if (mNbFoncIn==mNbFoncMax) 
              std::cout << mNbFoncIn ;
           else
              std::cout << "[" << mNbFoncIn << "," << mNbFoncMax << "]" ;

           std::cout << "  NbA=" ;
           if (mNbArgNum==mNbArgMax) 
              std::cout << mNbArgNum ;
           else
              std::cout << "[" << mNbArgNum << "," << mNbArgMax << "]" ;
           std::cout << "\n";
}


cFilterImPolI::cFilterImPolI(tPtrCalcFF aCalc,int aNbFoncIn,int  aNbFoncMax,int aNbArgNum,int  aNbArgMax,const std::string & aPat,bool aChgCtx,const std::string & aCom) :
    mCalc      (aCalc),
    mNbFoncIn  (aNbFoncIn),
    mNbFoncMax (aNbFoncMax),
    mNbArgNum  (aNbArgNum),
    mNbArgMax  (aNbArgMax),
    mPat       ("("+aPat + ")"),
    mAutom     (mPat,10),
    mChgCtx    (aChgCtx),
    mCom       (aCom)
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



  //----------------------------------------------------------------
static Fonc_Num FPermut(cFilterImPolI & aFIPI,const cArgFilterPolI & anArg) 
{

    Fonc_Num aF = anArg.mVIn.at(0);
    std::vector<int> aVI ;
    FromString(aVI,anArg.mVArgs.at(0) );

    return aF.permut(aVI);
}
static cFilterImPolI  Opermut(FPermut,1,1,1,1,"permut",false,"permut F [i1 i2]");

  //----------------------------------------------------------------
static Fonc_Num FKProj(cFilterImPolI & aFIPI,const cArgFilterPolI & anArg) 
{

    Fonc_Num aF = anArg.mVIn.at(0);

    return aF.kth_proj(ToInt(EndStr(anArg.mNameIn) )); // (( anArg.mNameIn.substr(1,std::string::npos)));
}
static cFilterImPolI  OperKProj(FKProj,1,1,0,0,"v[0-9]+",false,"v0 F or  v1 F  or ...." );



  //----------------------------------------------------------------
static Fonc_Num FSetSymb(cFilterImPolI & aFIPI,const cArgFilterPolI & anArg) 
{
    Fonc_Num aFonc = anArg.mVIn.at(0);
    // Symb_FNum* aRes = anArg.mCtx->Add(EndStr(anArg.mNameIn),aFonc);  // ((anArg.mNameIn.substr(1,std::string::npos),aFonc);

    if (anArg.mVIn.size()==1) 
       return  *(anArg.mCtx->GetValSymb(EndStr(anArg.mNameIn)));

    return anArg.mVIn.at(1);
}
static cFilterImPolI  OperSetSymb(FSetSymb,1,2,0,0,"=[A-Z,a-z].*",false,"=toto F or (=toto F1 F2)");

  //----------------------------------------------------------------
static Fonc_Num FUseSymb(cFilterImPolI & aFIPI,const cArgFilterPolI & anArg) 
{
    Symb_FNum* aRes = anArg.mCtx->GetValSymb(EndStr(anArg.mNameIn)) ; // anArg.mNameIn.substr(1,std::string::npos));

    return  *aRes;
}
static cFilterImPolI  OperUseSymb(FUseSymb,0,0,0,0,"@[A-Z,a-z].*",false,"@toto");

  //----------------------------------------------------------------

static Fonc_Num FAssoc(cFilterImPolI & aFIPI,const cArgFilterPolI & anArg) 
{
  const OperAssocMixte & anOp =   *(OperAssocMixte::GetFromName(anArg.mNameIn));

   Fonc_Num aRes = anOp.opf(anArg.mVIn.at(0),anArg.mVIn.at(1));
   for (int aK=2 ; aK<int(anArg.mVIn.size()) ; aK++)
       aRes = anOp.opf(aRes,anArg.mVIn.at(aK));
   return aRes;
}

static cFilterImPolI  OperAssoc(FAssoc,2,10000,0,0,"\\*|\\+|max|min",false,"max F1 F2 or (max F1 F2 F3 ....)");


  //----------------------------------------------------------------

static Fonc_Num FVirgule(cFilterImPolI & aFIPI,const cArgFilterPolI & anArg) 
{

   Fonc_Num aRes = Virgule(anArg.mVIn.at(0),anArg.mVIn.at(1));
   for (int aK=2 ; aK<int(anArg.mVIn.size()) ; aK++)
       aRes = Virgule(aRes,anArg.mVIn.at(aK));
   return aRes;
}

static cFilterImPolI  OperVirgule(FVirgule,2,10000,0,0,",",false,", F1 F2 or (,  F1 F2 F3 ....)");


  //----------------------------------------------------------------

static Fonc_Num FOperIf(cFilterImPolI & aFIPI,const cArgFilterPolI & anArg) 
{
    Symb_FNum aTest = (anArg.mVIn.at(0) != 0);

    return aTest * anArg.mVIn.at(1) + (!aTest) * anArg.mVIn.at(2);
}

static cFilterImPolI  OperIf(FOperIf,3,3,0,0,"\\?",false,"? F1 F2 F3");
  //----------------------------------------------------------------

static Fonc_Num FOperBin(cFilterImPolI & aFIPI,const cArgFilterPolI & anArg) 
{
    tOperFuncBin  anOper = OperFuncBinaireFromName(anArg.mNameIn);

    return anOper(anArg.mVIn.at(0),anArg.mVIn.at(1));
}

static std::string TheStrOpB="-|/|pow|>=|>|<|<=|==|!=|&|&&|(\\|)|(\\|\\|)|\\^|%|mod|>>|<<|f1f2bn";
static cFilterImPolI  OperBin(FOperBin,2,2,0,0,TheStrOpB,false,"pow F1 F2");
  //----------------------------------------------------------------

static Fonc_Num FOperUn(cFilterImPolI & aFIPI,const cArgFilterPolI & anArg) 
{
    tOperFuncUnaire anOper = OperFuncUnaireFromName(anArg.mNameIn);

    return anOper(anArg.mVIn.at(0));
}

static std::string TheStrOpU="u-|~|!|signed_frac|ecart_frac|cos|sin|tan|log|log2|exp|square|cube|abs|atan|sqrt|erfcc|isbadnum";
static cFilterImPolI  OperUn(FOperUn,1,1,0,0,TheStrOpU,false,"cos F");


  //----------------------------------------------------------------
static Fonc_Num FTif(cFilterImPolI &,const cArgFilterPolI & anArg)
{
   return  Tiff_Im::StdConvGen(anArg.mNameIn,-1,true).in_proj();
}
// static cFilterImPolI  OperTif(FTif,0,0,0,0,".*\\.(tif|tiff|Tif|Tiff|TIF|TIFF|jpg|jpeg|Jpg|Jpeg|JPG|JPEG)",false);
static cFilterImPolI  OperTif(FTif,0,0,0,0,".*\\.(tif|tiff|jpg|jpeg|cr2|arw|png|pfm|pgm)",false,"MyFile.tif");

  //----------------------------------------------------------------

static Fonc_Num FCoord(cFilterImPolI & aPolI,const cArgFilterPolI & anArg)
{
// std::cout << "JJJJJ " << anArg.mNameIn << " == " << aPolI.mAutom.Match(anArg.mNameIn) << "\n";
   int aKC =0 ;
   if (anArg.mNameIn.size() == 1)
      aKC =  tolower(anArg.mNameIn[0])- 'x';
   else
      aKC=  ToInt(anArg.mNameIn.substr(1,std::string::npos));

   return  kth_coord(aKC);
}

static cFilterImPolI  OperCoord(FCoord,0,0,0,0,"x|y|z|x[0-9]+",false,"x or y oz or x0 x1 ....");



  //----------------------------------------------------------------

static Fonc_Num FDoubleCste(cFilterImPolI &,const cArgFilterPolI & anArg)
{
   return   Fonc_Num(ToDouble(anArg.mNameIn));
}
static cFilterImPolI  OperDoubleCste(FDoubleCste,0,0,0,0,"-?[0-9]+\\.[0-9]*([eE]-?[0-9]+)?",false,"3.14");

static Fonc_Num FIntCste(cFilterImPolI &,const cArgFilterPolI & anArg)
{
   return   Fonc_Num(ToInt(anArg.mNameIn));
}
static cFilterImPolI  OperIntCste(FIntCste,0,0,0,0,"-?[0-9]+",false,"222");

  //----------------------------------------------------------------

static Fonc_Num FDeriche(cFilterImPolI &,const cArgFilterPolI & anArg)
{
   return   deriche(anArg.mVIn.at(0) ,ToDouble(anArg.mVArgs.at(0)),20);
}

static cFilterImPolI  OperDeriche(FDeriche,1,1,1,1,"deriche",true,"deriche F a ; a=exposant  in e(-a|x|)");

  //----------------------------------------------------------------

static Fonc_Num FPolar(cFilterImPolI &,const cArgFilterPolI & anArg)
{
   return   Polar_Def_Opun::polar(anArg.mVIn.at(0),0);
}

static cFilterImPolI  OperPolar(FPolar,1,1,0,0,"polar",false,"polar F");



  //----------------------------------------------------------------

static Fonc_Num FExtinc(cFilterImPolI &,const cArgFilterPolI & anArg)
{
    int aD = (anArg.mVArgs.size() >=2) ? ToInt(anArg.mVArgs.at(1)) : 256;
    const Chamfer &  aChmf=  Chamfer::ChamferFromName(anArg.mVArgs.at(0));

    return extinc(anArg.mVIn.at(0),aChmf,aD);
}

static cFilterImPolI  OperExtinc(FExtinc,1,1,1,2,"extinc",true,"extinc F c d ; c=chamfer d=distance");


  //----------------------------------------------------------------

static Fonc_Num FCourbTgt(cFilterImPolI &,const cArgFilterPolI & anArg)
{
    double aExp = (anArg.mVArgs.size() >=1) ? ToDouble(anArg.mVArgs.at(0)) : 0.5;

    return courb_tgt(anArg.mVIn.at(0),aExp);
}

static cFilterImPolI  OperCourbTgt(FCourbTgt,1,1,0,1,"corner",true,"corner F P? ; F=func P=pow, def=0.5");

  //----------------------------------------------------------------

static Fonc_Num FNoise(cFilterImPolI &,const cArgFilterPolI & anArg,bool Gauss)
{
    int aNbArgTot = anArg.mVArgs.size() ;
    ELISE_ASSERT(aNbArgTot && (aNbArgTot%2==0),"Bad Nb Arg in FNoise");
    int aNbV = aNbArgTot / 2;

    std::vector<double> aVPds;
    std::vector<int> aVSz;
    for (int aK=0 ; aK < aNbV ; aK++)
    {
         aVPds.push_back(ToDouble(anArg.mVArgs.at(2*aK)));
         aVSz.push_back(ToInt(anArg.mVArgs.at(2*aK+1)));
    }


    return   Gauss ? gauss_noise_4(aVPds.data(),aVSz.data(),aNbV) : unif_noise_4(aVPds.data(),aVSz.data(),aNbV);
}

static Fonc_Num FGaussNoise(cFilterImPolI &aFP,const cArgFilterPolI & anArg)
{
   return FNoise(aFP,anArg,true);
}

static cFilterImPolI  OperGaussNoise(FGaussNoise,0,0,2,10000,"gauss_noise",true,"gauss_noise p1 s1 p2 s2 ....");



  //----------------------------------------------------------------

static Fonc_Num FMinSelfDiSym(cFilterImPolI &,const cArgFilterPolI & anArg)
{
    int aNbWin = ToInt(anArg.mVArgs.at(0)) ;
    double aNbVois = ToDouble(anArg.mVArgs.at(1)) ;
    double aD2Max = ElSquare(aNbVois);
    int aNbVI = round_up(aNbVois);
    double aExp=2; // Par defaut prop au carre de la dist
 
    Fonc_Num aFonc = anArg.mVIn.at(0);
    Fonc_Num aFoncRes = Fonc_Num(1e10);

    for (int aDx=-aNbVI ; aDx<=aNbVI ; aDx++)
    {
        for (int aDy=-aNbVI ; aDy<=aNbVI ; aDy++)
        {
            int aD2 = ElSquare(aDx) + ElSquare(aDy);
            if ((aD2!=0 ) && (aD2 <= aD2Max))
            {
               Fonc_Num aFDif = rect_som(Abs(aFonc-trans(aFonc,Pt2di(aDx,aDy))),aNbWin);
               aFDif = aFDif / pow(sqrt(aD2),aExp);
               aFoncRes = Min(aFoncRes,aFDif);
            }
        }
    }

    return aFoncRes;
}

static cFilterImPolI  OperMinSelfDiSym(FMinSelfDiSym,1,1,2,2,"msd",true,"Self dissymilarity  : MinSelfDiSym Fonc Vois SzW");



  //----------------------------------------------------------------

static Fonc_Num FEroDil(cFilterImPolI &,const cArgFilterPolI & anArg)
{
    const Chamfer &  aChmf=  Chamfer::ChamferFromName(anArg.mVArgs.at(0));
    int aD =  ToInt(anArg.mVArgs.at(1)) ;

    return (anArg.mNameIn=="erode") ? erod(anArg.mVIn.at(0),aChmf,aD) : dilat(anArg.mVIn.at(0),aChmf,aD);
}

static cFilterImPolI  OperEroDil(FEroDil,1,1,2,2,"erode|dilate",true,"erode F c d ; c=chamfer d=distance");

  //----------------------------------------------------------------

static Fonc_Num FCloseOpen(cFilterImPolI &,const cArgFilterPolI & anArg)
{
    const Chamfer &  aChmf=  Chamfer::ChamferFromName(anArg.mVArgs.at(0));
    int aD =  ToInt(anArg.mVArgs.at(1)) ;
    int aDelta = (anArg.mVArgs.size() >=3)  ? ToInt(anArg.mVArgs.at(2)) : 0 ;

    return (anArg.mNameIn=="open") ? open(anArg.mVIn.at(0),aChmf,aD,aDelta) : close(anArg.mVIn.at(0),aChmf,aD,aDelta);
}

static cFilterImPolI  OperCloseOpen(FCloseOpen,1,1,2,3,"open|close",true,"(open F  c d1 d2) ; c=chamfer d1,d2=distance");

  //----------------------------------------------------------------

static Fonc_Num FMoy(cFilterImPolI &,const cArgFilterPolI & anArg)
{
    int aNbV =  ToInt(anArg.mVArgs.at(0)) ;
    int aNbIter =  (anArg.mVArgs.size()>=2) ? ToInt(anArg.mVArgs.at(1)) : 1 ;

    Fonc_Num aRes = Rconv(anArg.mVIn.at(0));
    for (int  aK=0 ; aK<aNbIter ; aK++)
    {
        aRes = rect_som(aRes,aNbV) / ElSquare(1.0+2.0*aNbV);
    }

    return aRes;
}

static cFilterImPolI  OperMoy(FMoy,1,1,1,2,"moy",true,"(moy F SzW NbIter)");

  //----------------------------------------------------------------


static Fonc_Num FMoyPond(cFilterImPolI &,const cArgFilterPolI & anArg)
{
    int aNbVx =  ToInt(anArg.mVArgs.at(0)) ;
    int aNbVy =  ToInt(anArg.mVArgs.at(1)) ;

    ELISE_ASSERT( int(anArg.mVArgs.size())==(2+aNbVx*aNbVy),"FMoyPond bad nb arg");
    Im2D_REAL8 anIm(aNbVx,aNbVy);
    int aCpt=2;
    double aSom = 0.0;
    for (int aKx=0 ; aKx<aNbVx ; aKx++)
    {
       for (int aKy=0 ; aKy<aNbVy ; aKy++)
       {
          double aV= ToDouble(anArg.mVArgs.at(aCpt));
          aSom += aV;
          anIm.SetR(Pt2di(aKx,aKy),aV);
          aCpt++;
       }
    }
    ELISE_COPY(anIm.all_pts(),anIm.in()/aSom,anIm.out());
   
    return som_masq(Rconv(anArg.mVIn.at(0)),anIm);
}

static cFilterImPolI  OperMoyPond(FMoyPond,1,1,2,500,"moyp",true,"(moyp F 3 3 1 2 1 2 4 2 1 2 1)");




  //----------------------------------------------------------------

static Fonc_Num FMaxMin(cFilterImPolI &,const cArgFilterPolI & anArg)
{
    int aNbV =  ToInt(anArg.mVArgs.at(0)) ;
    int aNbIter =  (anArg.mVArgs.size()>=2) ? ToInt(anArg.mVArgs.at(1)) : 1 ;

    Fonc_Num aRes = Rconv(anArg.mVIn.at(0));
    for (int  aK=0 ; aK<aNbIter ; aK++)
    {
        if      (anArg.mNameIn=="maxv") aRes =  rect_max(aRes,aNbV);
        else if (anArg.mNameIn=="minv") aRes =  rect_min(aRes,aNbV);
        else {ELISE_ASSERT(false,"Bas name in FMaxMin");}
    }

    return aRes;
}

static cFilterImPolI  OperMaxMin(FMaxMin,1,1,1,2,"maxv|minv",true,"(maxv SzW NbIter)");




  //----------------------------------------------------------------

static Fonc_Num FMedian(cFilterImPolI & aFIPI,const cArgFilterPolI & anArg) 
{
   int aNbV =  ToInt(anArg.mVArgs.at(0)) ;
   int aNbIter =  (anArg.mVArgs.size()>=2) ? ToInt(anArg.mVArgs.at(1)) : 1 ;
   Fonc_Num aRes = anArg.mVIn.at(0);

   for (int  aK=0 ; aK<aNbIter ; aK++)
       aRes = MedianBySort(aRes,aNbV);

    return aRes;
}
static cFilterImPolI  OperMed(FMedian,1,1,1,2,"median",true,"(median F SzW NbIter)");


  //----------------------------------------------------------------
//        0    1  2    3    4
//  kth F 0.5  Sz VMin VMax Nbal  (NbIter=1) (Sz.y=Sz=Sz.x)

static Fonc_Num FIKth(cFilterImPolI & aFIPI,const cArgFilterPolI & anArg) 
{
   int aNbArg = anArg.mVArgs.size();
   double   aProp = ToDouble(anArg.mVArgs.at(0));
   int      aSzX  =    ToInt(anArg.mVArgs.at(1));
   double   aVMin = ToDouble(anArg.mVArgs.at(2));
   double   aVMax = ToDouble(anArg.mVArgs.at(3));
   int   aNbVal   =     ToInt(anArg.mVArgs.at(4));
   int      aNbIter = (aNbArg >=6) ? ToInt(anArg.mVArgs.at(5)) : 1;
   int      aSzY    = (aNbArg >=7) ? ToInt(anArg.mVArgs.at(6)) : aSzX;
   
 
   Fonc_Num aRes  = anArg.mVIn.at(0);
   aRes = Max(0,Min(aNbVal-1,round_ni(aNbVal *(aRes-aVMin) /(aVMax-aVMin))));
   
   int aNbVois =  (1+2*aSzX) * (1+2*aSzY);

   for (int  aK=0 ; aK<aNbIter ; aK++)
       aRes = rect_kth(aRes,round_ni(aProp*(aNbVois-1)),Pt2di(aSzX,aSzY),aNbVal);

    aRes = aVMin + aRes * (aVMax-aVMin) / aNbVal;
    return aRes;
}
static cFilterImPolI  OperIKth(FIKth,1,1,5,7,"ikth",true,"(ikth F Prop SzW VMin VMax NbDisc ?NbIter ?SzWy)");


  //----------------------------------------------------------------

/*
static Fonc_Num FSobel(cFilterImPolI & aFIPI,const cArgFilterPolI & anArg) 
{
    Fonc_Num aFonc = anArg.mVIn.at(0);
    return sobel(aFonc);
}
static cFilterImPolI  OperSobel(FSobel,1,1,0,0,"sobel",true);
*/

  //----------------------------------------------------------------

static Fonc_Num FTrans(cFilterImPolI & aFIPI,const cArgFilterPolI & anArg) 
{
   int aTX =  ToInt(anArg.mVArgs.at(0)) ;
   int aTY =  ToInt(anArg.mVArgs.at(1)) ;
   Fonc_Num aFonc = anArg.mVIn.at(0);

    return trans(aFonc,Pt2di(aTX,aTY));
}
static cFilterImPolI  OperTrans(FTrans,1,1,2,2,"trans",true,"trans F dx dy");

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
         aRes.push_back(&OperDeriche);
         aRes.push_back(&OperIntCste);
         aRes.push_back(&OperDoubleCste);
         aRes.push_back(&OperPolar);
         aRes.push_back(&OperExtinc);
         aRes.push_back(&OperCourbTgt);
         aRes.push_back(&OperGaussNoise);
         aRes.push_back(&OperMinSelfDiSym);
         aRes.push_back(&Opermut);
         aRes.push_back(&OperSetSymb);
         aRes.push_back(&OperUseSymb);
         aRes.push_back(&OperKProj);
         aRes.push_back(&OperVirgule);
         aRes.push_back(&OperIf);
         aRes.push_back(&OperEroDil);
         aRes.push_back(&OperCloseOpen);
         aRes.push_back(&OperMoy);
         aRes.push_back(&OperMed);
         aRes.push_back(&OperTrans);
         aRes.push_back(&OperIKth);
         aRes.push_back(&OperMaxMin);
         aRes.push_back(&OperMoyPond);
         // aRes.push_back(&OperSobel);
    }

    return aRes;
}

//=============================


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

cResFilterPolI RecParseStrFNPolI(tCPtr & aStr,cCtxtFoncPolI * aCtx)
{
    cCtxtFoncPolI * aCtxFils = aCtx;
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
    std::string  aIdSymb = StrToLower(aSymb);


    std::vector<cFilterImPolI *>  aVPol =  VPolI();

    for (int aK=0 ; aK<int(aVPol.size()) ; aK++)
    {
        cFilterImPolI & aPolI = *(aVPol[aK]);
// std::cout << "HHHHH " << aPolI.mAutom.NameExpr() << " " << aIdSymb << "\n";
        if (aPolI.mAutom.Match(aIdSymb))
        {
            if (aPolI.mChgCtx)
            {
               aCtxFils = aCtx->Dup();
            }
            int aNbFonc =  aParOuv ? aPolI.mNbFoncMax : aPolI.mNbFoncIn ;

            cArgFilterPolI anArg(aSymb,aCtx);
            bool GotParFerm=false;
            for (int aK=0 ; (aK<aNbFonc) && (!GotParFerm)  ; aK++)
            {
                 cResFilterPolI aRFPI = RecParseStrFNPolI(aStr,aCtxFils);
                 if (aRFPI.mSymbSpec==")")
                 {
                     GotParFerm =true;
                 }
                 else
                 {
                    anArg.mVIn.push_back(aRFPI.mFonc);
                    anArg.mBox = UnionBoxPtrWithDel(anArg.mBox,aRFPI.mBox);
                    if ((&aPolI == &OperSetSymb) && (aK==0))
                    {
                       aCtx->Add(EndStr(aSymb),aRFPI.mFonc);
                    }
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
            if (aCtxFils!=aCtx)
               aCtx->Herit(aCtxFils);

            return  cResFilterPolI(aPolI.mCalc(aPolI,anArg),anArg.mBox);
        }
        else
        {
             // std::cout << "SYMB NOT MATCHED= " << aIdSymb << "\n";
        }
    }

    std::cout << "For symb=[" << aSymb << "]\n";
    ELISE_ASSERT(false,"could not recognize symbol");

    return * ((cResFilterPolI *) 0);
}

cResFilterPolI GlobParseStrFNPolI(tCPtr & aStr)
{
    aStrGlob = aStr;
    cCtxtFoncPolI * aCtx = new cCtxtFoncPolI;

    cResFilterPolI aRes = RecParseStrFNPolI(aStr,aCtx);

    while (*aStr && isspace(*aStr))
    {
       aStr++;
    }

    if (aStr!=std::string(""))
    {
       std::cout << "remaining string=[" << aStr << "]\n";
       ELISE_ASSERT(false,"did not read all string");
    }
    
    return aRes;
}

void NirupActionOnHelp(int argc,char ** argv)
{
     // std::vector<cFilterImPolI *>  aVP =  VPolI();
     for  (const auto & aP :  VPolI())
     {
        aP->Show();
     }
}



int Nikrup_main(int argc,char ** argv)
{
    std::string anExpr;
    std::string aNameOut;
    Box2di       aBoxOut;
    // GenIm::type_el aType=  GenIm::real4;
    std::string aNameTypeOut = "real4";
    int aNbChan;

    TheActionOnHelp = NirupActionOnHelp;

/*
for (int ak=0 ; ak<argc ; ak++)
    std::cout << "NNnn [" << argv[ak] << "]\n";
*/

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

          L_Arg_Opt_Tiff aL;
          aL = aL+Arg_Tiff(Tiff_Im::ANoStrip());

          Tiff_Im aTifOut
                 (
                    aNameK.c_str(),
                    aSzOut,
                    aType,
                    Tiff_Im::No_Compr,
                    aPIT,
                    aL
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

     Nikrup_Banniere();
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


int TournIm_main(int argc,char ** argv)
{
    std::string aNameIm;
    std::string aNameOut;
    int  aNumGeom = 1;
    ElInitArgMain
    (
         argc,argv,
         LArgMain()  << EAMC(aNameIm,"Name of Input image", eSAM_IsExistFile),
         LArgMain()
                     << EAM(aNumGeom,"NumGeom",true,"0=>Id,1=>90(def), 2=>180,3=>270,[4-7]=>Sym", eSAM_NoInit)
                     << EAM(aNameOut,"Out",true,"Destination")
    );

    Tiff_Im aTif =  Tiff_Im::StdConvGen(aNameIm,-1,true);
    std::vector<Im2DGen *>  aVIm = aTif.ReadVecOfIm();
    Pt2di aSz = aVIm[0]->sz();
    std::cout << "SZ IN " << aSz << "\n";

    Fonc_Num aFTrans = Virgule(FY,aSz.y-1-FX);
    std::string aPref = "T90-";

    if (aNumGeom==0) 
    {
        aPref = "T0-";
        aFTrans = Virgule(FX,FY);
    }
    else if (aNumGeom==1) 
    {
        aFTrans = Virgule(FY,aSz.y-1-FX);
    }
    else if (aNumGeom==2)
    {
        aPref = "T180-";
        aFTrans = Virgule(aSz.x-1-FX,aSz.y-1-FY);
    }
    else if (aNumGeom==3)
    {
        aPref = "T270-";
        aFTrans = Virgule(aSz.x-1-FY,FX);
    }
    else if (aNumGeom==4) 
    {
        aPref = "SymX-";
        aFTrans = Virgule(aSz.x-1-FX,FY);
    }
    else if (aNumGeom==6) 
    {
        aPref = "SymY-";
        aFTrans = Virgule(FX,aSz.y-1-FY);
    }
    else
    {
        ELISE_ASSERT(false,"Unhandled value for NumGeom");
    }

    if (!EAMIsInit(&aNameOut))
        aNameOut = DirOfFile(aNameIm) + aPref+ StdPrefix(NameWithoutDir(aNameIm))+".tif";

    L_Arg_Opt_Tiff aLarg;
    aLarg = aLarg+  Arg_Tiff(Tiff_Im::ANoStrip());
    Tiff_Im aTifOut
            (
                aNameOut.c_str(),
                ((aNumGeom%2)==1)  ? Pt2di(aSz.y,aSz.x) : Pt2di(aSz.x,aSz.y),
                aTif.type_el(),
                Tiff_Im::No_Compr,
                aTif.phot_interp(),
                aLarg
            );
    std::cout << "SZ OUT " << aTifOut.sz() << "\n";

    Fonc_Num aF;
    int aKIm=0;



    // aFTrans  = Virgule(FY,aSz.y-1-FX);
    for (auto aI : aVIm)
    {

        Fonc_Num aNewF = aI->in()[aFTrans];
        aF = (aKIm) ? Virgule(aF,aNewF) : aNewF;
        aKIm++;
    }
/*
    int aX0,aX1,aY0,aY1;
    ELISE_COPY(
         aTifOut.all_pts(),
         aFTrans, // Virgule(FY,aSz.x-1-FX),
         Virgule(VMin(aX0)|VMax(aX1),VMin(aY0)|VMax(aY1))
    );
    std::cout << "XXX " << aX0 << " " << aX1 << ";; Y " << aY0 << " " << aY1 << "\n";
*/
    ELISE_COPY(aTifOut.all_pts(),aF,aTifOut.out());

    return EXIT_SUCCESS;
}

/**********  Test a bunch of idea about filters *******/

int DivFilters_main(int argc,char ** argv)
{
    std::string aNameIn;
    std::vector<std::string> aParamSupMoinsInf;

    ElInitArgMain
    (
         argc,argv,
         LArgMain()  << EAMC(aNameIn,"Name of Input image", eSAM_IsExistFile),
         LArgMain()  << EAM(aParamSupMoinsInf,"SMI",true,"Sup Moins Inf [SzW]", eSAM_NoInit)
    );

    if (!MMVisualMode)
    {
        Im2D<float,double> aIm=  Im2D<float,double>::FromFileStd(aNameIn);
        TIm2D<float,double>  aDIm(aIm);
        Pt2di aSzIm = aIm.sz();

        if (EAMIsInit(&aParamSupMoinsInf))
        {
             Im2D<float,double> aImNbInfSup(aSzIm.x,aSzIm.y);
             int aSzW;  FromString(aSzW,aParamSupMoinsInf.at(0));
             Pt2di aP;
             for (aP.y=aSzW ; aP.y<aSzIm.y-aSzW  ; aP.y++)
             {
                 for (aP.x=aSzW ; aP.x<aSzIm.x-aSzW  ; aP.x++)
                 {
                      int aNbSup=0;
                      int aNbInf=0;
                      float aV0 = aDIm.get(aP);
                      for (int aDy=-aSzW; aDy<=aSzW ; aDy++)
                      {
                          for (int aDx=-aSzW; aDx<=aSzW ; aDx++)
                          {
                              if ((aDx!=0) || (aDy!=0))
                              {
                                  Pt2di aDP(aDx,aDy);
                                  Pt2di aQ = aP+aDP;
                                  float aV1 = aDIm.get(aQ) + aDx/10.0 + aDy/100.0;
                                  if (aV1>aV0)
                                  {
                                      aNbSup++;
                                  }
                                  else
                                  {
                                      aNbInf++;
                                  }
                              }
                          }
                      }
                      double aProp = (aNbInf-aNbSup)/double(aNbInf+aNbSup);
                      double V5 = (aProp>0) ? 0.5 : -0.5;
                      aImNbInfSup.SetR(aP,std::abs(aProp-V5));
                 }
             }
             Tiff_Im::CreateFromIm(aImNbInfSup,"InfSup-"+StdPrefix(aNameIn)+".tif");
        }
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
