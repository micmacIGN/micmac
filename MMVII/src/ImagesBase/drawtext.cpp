#include "MMVII_Image2D.h"

using namespace MMVII;

// Taken from MicMac v1

namespace { // Private

// We will make it public later if needed
class cBitmFont
{
public :
    static  cBitmFont & BasicFont_10x8();
    static  cBitmFont & FontCodedTarget();
    virtual cIm2D<tU_INT1> ImChar(char) = 0;
    virtual ~cBitmFont() {}

    cIm2D<tU_INT1> BasicImageString(const std::string & ,int aSpace);
    cIm2D<tU_INT1> MultiLineImageString(const std::string & aStrInit,const cPt2di& aSpace,cPt2di aRab,int Centering);
};

} // namespace Private

namespace MMVII {
// The public API ...
cIm2D<tU_INT1> ImageOfString_10x8(const std::string & aStr ,int aSpace)
{
    return cBitmFont::BasicFont_10x8().BasicImageString(aStr,aSpace);
}

cIm2D<tU_INT1> ImageOfString_DCT(const std::string & aStr ,int aSpace)
{
    return  cBitmFont::FontCodedTarget().BasicImageString(aStr,aSpace);
}

cIm2D<tU_INT1> MultiLineImageOfString_10x8(const std::string & aStrInit,const cPt2di& aSpace,cPt2di aRab,int Centering)
{
    return cBitmFont::BasicFont_10x8().MultiLineImageString(aStrInit,aSpace,aRab,Centering);
}

cIm2D<tU_INT1> MultiLineImageOfString_DCT(const std::string & aStrInit,const cPt2di& aSpace,cPt2di aRab,int Centering)
{
    return cBitmFont::FontCodedTarget().MultiLineImageString(aStrInit,aSpace,aRab,Centering);
}

} // namespace MMVII

namespace {     // private

std::list<std::string> SplitStrings(const std::string & aCppStr)
{
    std::list<std::string> aRes;
    const char *  aStrInit = aCppStr.c_str();
    const char * aStr0 = aStrInit;
    bool Cont = (*aStr0!=0);

    while (Cont)
    {
        const char * aStr1 = aStr0;
        while ((*aStr1!=0) &&  (*aStr1 !='\n'))
            aStr1++;
        aRes.push_back(std::string(aStr0,aStr1));

        if (*aStr1==0)
            Cont= false;
        else
            aStr0 = aStr1+1;
    }

    return aRes;
}


/******************************************************/
/*                                                    */
/*              cImplemBitmFont                       */
/*                                                    */
/******************************************************/

class cImplemBitmFont : public cBitmFont, public cObj2DelAtEnd
{
public :

    cImplemBitmFont(const cPt2di& aSz,const std::string & aName);
    virtual ~cImplemBitmFont();
    void AddChar(char,const char*);

private :
    virtual cIm2D<tU_INT1> ImChar(char) ;


    std::map<char, cIm2D<tU_INT1> *> mDico;

    cPt2di mSz;
    std::string mName;
};

cImplemBitmFont::cImplemBitmFont(const cPt2di& aSz,const std::string & aName) :
    mSz   (aSz),
    mName (aName)
{
}

cImplemBitmFont::~cImplemBitmFont()
{
    for (auto& it: mDico)
    {
        delete it.second;
    }
}


cIm2D<tU_INT1> cImplemBitmFont::ImChar(char aChar)
{
    auto aResIt = mDico.find(aChar);
    MMVII_INTERNAL_ASSERT_strong(aResIt != mDico.end(),  std::string("cImplemBitmFont::ImChar cannot Get Im for font ") + mName + " With Char [" + aChar + "]");
    return *aResIt->second;
}

void cImplemBitmFont::AddChar(char anEntry,const char* aStr)
{
    MMVII_INTERNAL_ASSERT_strong(
        strlen(aStr) == (size_t)(mSz.x()*mSz.y()),
        std::string("cImplemBitmFont::AddChar bad size for char=[") + anEntry + "] ASCII=" + std::to_string((int)anEntry)
        );
    auto aVal = new cIm2D<tU_INT1> (mSz);
    mDico[anEntry] = aVal;

    for (const auto& aPt : aVal->DIm()) {
        aVal->DIm().SetV(aPt,*(aStr++) == '#' ? 255 : 0);
    }
}

/******************************************************/
/*                                                    */
/*              cBitmFont                             */
/*                                                    */
/******************************************************/

cIm2D<tU_INT1> cBitmFont::BasicImageString(const std::string & aStr,int aSpace)
{
    bool  isHor = (aSpace>=0);
    aSpace = std::abs(aSpace);

    if (aStr=="")
    {
        return  ImChar(' ');
    }
    cPt2di aSzRes(0,0);
    cPt2di aSzIm (0,0);

    for (const char * aC = aStr.c_str(); (*aC) ; aC++)
    {
        cIm2D<tU_INT1> anIm = ImChar(*aC);
        aSzIm = anIm.DIm().Sz();
        if (aSzRes.x()==0)
        {
            aSzRes = aSzIm;
        }
        else
        {
            if (isHor)
            {
                aSzRes.x() += aSzIm.x() + aSpace;
                MMVII_INTERNAL_ASSERT_strong(aSzRes.y() == aSzIm.y(),"cBitmFont::BasicImageString: ImageString Height variable for font");
            }
            else
            {
                aSzRes.y() += aSzIm.y() + aSpace;
                MMVII_INTERNAL_ASSERT_strong(aSzRes.x()==aSzIm.x(),"cBitmFont::BasicImageString: ImageString Width variable for font");
            }
        }
    }

    cIm2D<tU_INT1> aRes(aSzRes, nullptr, eModeInitImage::eMIA_Null);
    cPt2di anOffset(0,0);

    for (const char * aC = aStr.c_str(); (*aC) ; aC++)
    {
        cIm2D<tU_INT1> anIm = ImChar(*aC);
        const cPt2di& aSzIm = anIm.DIm().Sz();
        auto aResOffset = anOffset + aRes.DIm().P0() - anIm.DIm().P0();
        for (const auto& aPt : anIm.DIm())
        {
            aRes.DIm().SetV(aPt + aResOffset, anIm.DIm().GetV(aPt));
        }
        if (isHor)
            anOffset.x() += aSzIm.x() + aSpace;
        else
            anOffset.y() += aSzIm.y() + aSpace;
    }

    return aRes;
}

cIm2D<tU_INT1> cBitmFont::MultiLineImageString(const std::string & aStrInit,const cPt2di& aSpace,cPt2di aRab,int Centering)
{
    std::list<std::string> aLStr =  SplitStrings(aStrInit);

    std::list<cIm2D<tU_INT1>> aLIm;

    int aXMax=0;
    int aSomY = 0;

    std::list<cIm2D<tU_INT1>> aLB;
    for (std::list<std::string>::const_iterator itS=aLStr.begin() ;itS!=aLStr.end() ; itS++)
    {
        cIm2D<tU_INT1>  anIm = BasicImageString(*itS,aSpace.x());
        cPt2di aSz = anIm.DIm().Sz();
        aXMax = std::max(aXMax,aSz.x());
        aSomY += aSz.y();
        aLB.push_back(anIm);
    }
    aSomY += aSpace.y()*(int)(aLStr.size()-1);
    if (aRab.x()<0)  // Alors Rab est une taille totale
    {
        aRab = - aRab;
        aRab = (aRab- cPt2di(aXMax,aSomY))/2;
    }

    cPt2di aSzRes = (aRab*2) + cPt2di(aXMax,aSomY);

    cIm2D<tU_INT1>  aRes(aSzRes, nullptr, eModeInitImage::eMIA_Null);

    cPt2di aP0(0,aRab.y());

    for (std::list<cIm2D<tU_INT1> >::const_iterator itB=aLB.begin() ;itB!=aLB.end() ; itB++)
    {

        cIm2D<tU_INT1>  anIm = *itB;
        cPt2di aSz =  anIm.DIm().Sz();

        aP0.x() = aRab.x(); // cas -1
        if (Centering==0)
            aP0.x() = (aSzRes.x()-aSz.x()) /2;
        else if (Centering>0)
            aP0.x() = (aSzRes.x()-aSz.x()) - aRab.x();

        auto aResOffset = aP0 + aRes.DIm().P0() - anIm.DIm().P0();
        for (const auto& aPt : anIm.DIm())
        {
            aRes.DIm().SetV(aPt + aResOffset, anIm.DIm().GetV(aPt));
        }
        aP0.y() += aSz.y() + aSpace.y();
    }

    return aRes;
}


cBitmFont & cBitmFont::FontCodedTarget()
{
    static cImplemBitmFont* theFontCodedTarget = 0;
    if (theFontCodedTarget !=0)
        return *theFontCodedTarget;

    theFontCodedTarget = new cImplemBitmFont(cPt2di(11,11),"DCT");
    cMMVII_Appli::CurrentAppli().AddObj2DelAtEnd(theFontCodedTarget);

    theFontCodedTarget->AddChar
        (
            '0',
            "..####....."
            ".#....#...."
            ".#....#...."
            ".#....#...."
            ".#....#...."
            ".#....#...."
            ".#....#...."
            "..####....."
            "..........."
            "..........."
            "..........."
            );

    theFontCodedTarget->AddChar
        (
            '1',
            "....#......"
            "...##......"
            "..#.#......"
            ".#..#......"
            "....#......"
            "....#......"
            "....#......"
            "....#......"
            "..........."
            "..........."
            "..........."
            );

    theFontCodedTarget->AddChar
        (
            '2',
            ".###......."
            "#...#......"
            "#....#....."
            "....#......"
            "...#......."
            "..#........"
            "######....."
            "..........."
            "..........."
            "..........."
            "..........."
            );

    theFontCodedTarget->AddChar
        (
            '3',
            "..##......."
            ".#..#......"
            "....#......"
            "..##......."
            ".....#....."
            ".#...#....."
            "..###......"
            "..........."
            "..........."
            "..........."
            "..........."
            );

    theFontCodedTarget->AddChar
        (
            '4',
            "....#......"
            "...#......."
            "..#........"
            ".#........."
            "#...#......"
            "######....."
            "....#......"
            "....#......"
            "..........."
            "..........."
            "..........."
            );

    theFontCodedTarget->AddChar
        (
            '5',
            ".#####....."
            ".#........."
            "#.........."
            "####......."
            "....#......"
            ".....#....."
            ".....#....."
            "#####......"
            "..........."
            "..........."
            "..........."
            );

    theFontCodedTarget->AddChar
        (
            '6',
            "...###....."
            "..#...#...."
            " #........."
            ".####......"
            "#.....#...."
            "#.....#...."
            ".#####....."
            "..........."
            "..........."
            "..........."
            "..........."
            );

    theFontCodedTarget->AddChar
        (
            '7',
            ".#####....."
            ".....#....."
            "....#......"
            "..####....."
            "...#......."
            "...#......."
            "..#........"
            "..........."
            "..........."
            "..........."
            "..........."
            );

    theFontCodedTarget->AddChar
        (
            '8',
            "...###....."
            "..#...#...."
            "..#...#...."
            "...###....."
            "..#...#...."
            ".#.....#..."
            ".#....#...."
            "..####....."
            "..........."
            "..........."
            "..........."
            );

    theFontCodedTarget->AddChar
        (
            '9',
            "..#####...."
            ".#.....#..."
            ".#....##..."
            "..####.#..."
            ".......#..."
            ".....#....."
            "..###......"
            "..........."
            "..........."
            "..........."
            "..........."
            );


    theFontCodedTarget->AddChar
        (
            'A',
            "...###....."
            "..#...#...."
            ".#.....#..."
            ".#######..."
            ".#.....#..."
            ".#.....#..."
            ".#.....#..."
            "..........."
            "..........."
            "..........."
            "..........."
            );

    theFontCodedTarget->AddChar
        (
            'B',
            ".####......"
            ".#...#....."
            ".#..#......"
            ".######...."
            ".#.....#..."
            ".#.....#..."
            ".######...."
            "..........."
            "..........."
            "..........."
            "..........."
            );

    theFontCodedTarget->AddChar
        (
            'C',
            "...####...."
            ". #...#...."
            ".#........."
            ".#........."
            "#.........."
            "#....#....."
            ".####......"
            "..........."
            "..........."
            "..........."
            "..........."
            );

    theFontCodedTarget->AddChar
        (
            'D',
            "..####....."
            "..#...#...."
            "..#....#..."
            ".#....#...."
            ".#....#...."
            ".#..##....."
            "####......."
            "..........."
            "..........."
            "..........."
            "..........."
            );

    theFontCodedTarget->AddChar
        (
            'E',
            "..#####...."
            "..#........"
            ".#........."
            ".###......."
            ".#........."
            "#.........."
            "#####......"
            "..........."
            "..........."
            "..........."
            "..........."
            );
    theFontCodedTarget->AddChar
        (
            'F',
            "..######..."
            "..#........"
            "..#........"
            "..###......"
            ".#........."
            ".#........."
            ".#........."
            "..........."
            "..........."
            "..........."
            "..........."
            );

    return *theFontCodedTarget;
}


cBitmFont & cBitmFont::BasicFont_10x8()
{
    static cImplemBitmFont *theFont_10x8 = 0;
    if (theFont_10x8 !=0)
        return *theFont_10x8;

    theFont_10x8 = new cImplemBitmFont(cPt2di(10,10),"El_10x8");
    cMMVII_Appli::CurrentAppli().AddObj2DelAtEnd(theFont_10x8);

    theFont_10x8->AddChar
        (
            '_',
            ".........."
            ".........."
            ".........."
            ".........."
            ".........."
            ".........."
            ".........."
            "..#######."

            ".........."
            ".........."
            );
    theFont_10x8->AddChar
        (
            ' ',
            ".........."
            ".........."
            ".........."
            ".........."
            ".........."
            ".........."
            ".........."
            ".........."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            '.',
            ".........."
            ".........."
            ".........."
            ".........."
            ".........."
            ".........."
            "....##...."
            "....##...."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            '0',
            "...####..."
            "..#....#.."
            "..#....#.."
            "..#....#.."
            "..#....#.."
            "..#....#.."
            "..#....#.."
            "...####..."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            '1',
            "......#..."
            ".....##..."
            "....#.#..."
            "...#..#..."
            "......#..."
            "......#..."
            "......#..."
            "....####.."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            '2',
            "...###...."
            "..#...#..."
            "..#....#.."
            "......#..."
            ".....#...."
            "....#....."
            "...#......"
            "..######.."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            '3',
            "....##...."
            "...#..#..."
            "......#..."
            "....##...."
            "......#..."
            ".......#.."
            "...#...#.."
            "....###..."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            '4',
            "......#..."
            ".....#...."
            "....#....."
            "...#......"
            "..#...#..."
            "..######.."
            "......#..."
            "......#..."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            '5',
            "..#####..."
            "..#......."
            "..#......."
            "..####...."
            "......#..."
            ".......#.."
            ".......#.."
            "..#####..."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            '6',
            ".....###.."
            "....#....."
            ".. #......"
            "..#......."
            "..#.####.."
            "..##....#."
            "..#.....#."
            "...#####.."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            '7',
            "...#####.."
            ".......#.."
            "......#..."
            "....####.."
            ".....#...."
            ".....#...."
            "....#....."
            "....#....."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            '8',
            "...####..."
            "..#....#.."
            "..#....#.."
            "...####..."
            "..#....#.."
            ".#......#."
            ".#......#."
            "..######.."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            '9',
            "..#####..."
            ".#.....#.."
            ".#....##.."
            "..####.#.."
            ".......#.."
            "......#..."
            ".....#...."
            "..###....."

            ".........."
            ".........."
            );


    theFont_10x8->AddChar
        (
            'A',
            "...####..."
            "..#....#.."
            ".#......#."
            ".#......#."
            ".########."
            ".#......#."
            ".#......#."
            ".#......#."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'B',
            ".#####...."
            ".#....#..."
            ".#....#..."
            ".#####...."
            ".#.....#.."
            ".#......#."
            ".#.....#.."
            ".######..."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'C',
            "...####..."
            "..#....#.."
            "..#......."
            "..#......."
            "..#......."
            "..#......."
            "..#....#.."
            "...####..."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'D',
            "..####...."
            "..#...#..."
            "..#....#.."
            "..#....#.."
            "..#....#.."
            "..#....#.."
            "..#...#..."
            "..####...."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'E',
            "..######.."
            "..#......."
            "..#......."
            "..#####..."
            "..#......."
            "..#......."
            "..#......."
            "..######.."

            ".........."
            ".........."
            );
    theFont_10x8->AddChar
        (
            'F',
            "..######.."
            "..#......."
            "..#......."
            "..#####..."
            "..#......."
            "..#......."
            "..#......."
            "..#......."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'G',
            "...####..."
            "..#....#.."
            "..#......."
            "..#......."
            "..#.. ###."
            "..#....#.."
            "..#....#.."
            "...####..."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'H',
            "..#....#.."
            "..#....#.."
            "..#....#.."
            "..######.."
            "..#....#.."
            "..#....#.."
            "..#....#.."
            "..#....#.."

            ".........."
            ".........."
            );


    theFont_10x8->AddChar
        (
            'I',
            "..#######."
            ".....#...."
            ".....#...."
            ".....#...."
            ".....#...."
            ".....#...."
            ".....#...."
            "..#######."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'J',
            "..#######."
            ".......#.."
            ".......#.."
            ".......#.."
            ".......#.."
            ".......#.."
            "..#...#..."
            "...###...."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'K',
            "..#....#.."
            "..#...#..."
            "..#..#...."
            "..##......"
            "..#..#...."
            "..#...#..."
            "..#....#.."
            "..#.....#."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'L',
            "..#......."
            "..#......."
            "..#......."
            "..#......."
            "..#......."
            "..#......."
            "..#......."
            "..#######."

            ".........."
            ".........."
            );




    theFont_10x8->AddChar
        (
            'M',
            "..#.....#."
            "..##...##."
            "..#.#.#.#."
            "..#..#..#."
            "..#.....#."
            "..#.....#."
            "..#.....#."
            "..#.....#."

            ".........."
            ".........."
            );


    theFont_10x8->AddChar
        (
            'N',
            "..#.....#."
            "..##....#."
            "..#.#...#."
            "..#..#..#."
            "..#...#.#."
            "..#....##."
            "..#.....#."
            "..#.....#."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'O',
            "...####..."
            "..#....#.."
            ".#......#."
            ".#......#."
            ".#......#."
            ".#......#."
            "..#....#.."
            "...####..."
            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'P',
            "..####...."
            "..#...##.."
            "..#....#.."
            "..#...##.."
            "..####...."
            "..#......."
            "..#......."
            "..#......."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'Q',
            "...####..."
            "..#....#.."
            ".#......#."
            ".#......#."
            ".#......#."
            ".#......#."
            "..#.#..#.."
            "...####..."
            ".....#...."
            "......#..."
            );


    theFont_10x8->AddChar
        (
            'R',
            "..#####..."
            "..#....#.."
            "..#....#.."
            "..#####..."
            "..#....#.."
            "..#.....#."
            "..#.....#."
            "..#.....#."

            ".........."
            ".........."
            );


    theFont_10x8->AddChar
        (
            'S',
            "...####..."
            "..#....#.."
            "..#......."
            "...#####.."
            "........#."
            "........#."
            "..#.....#."
            "...#####.."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'T',
            ".########."
            ".....#...."
            ".....#...."
            ".....#...."
            ".....#...."
            ".....#...."
            ".....#...."
            ".....#...."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'U',
            "..#.....#."
            "..#.....#."
            "..#.....#."
            "..#.....#."
            "..#.....#."
            "..#.....#."
            "..#....##."
            "...####.#."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'V',
            "..#.....#."
            "..#.....#."
            "..#.....#."
            "...#...#.."
            "...#...#.."
            "...#...#.."
            "....#.#..."
            ".....#...."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'W',
            "..#.....#."
            "..#.....#."
            "..#.....#."
            "..#.....#."
            "..#..#..#."
            "..#.#.#.#."
            "..##...##."
            "..#.....#."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'X',
            "..#.....#."
            "...#...#.."
            "....#.#..."
            ".....#...."
            "....#.#..."
            "...#...#.."
            "..#.....#."
            "..#.....#."

            ".........."
            ".........."
            );


    theFont_10x8->AddChar
        (
            'Y',
            "..#.....#."
            "...#...#.."
            "....#.#..."
            ".....#...."
            ".....#...."
            ".....#...."
            ".....#...."
            ".....#...."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'Z',
            ".########."
            ".#.....#.."
            "......#..."
            ".....#...."
            "....#....."
            "...#......"
            "..#.....#."
            ".########."

            ".........."
            ".........."
            );


    theFont_10x8->AddChar
        (
            'a',
            ".........."
            ".........."
            "...####..."
            ".......#.."
            "...#####.."
            "..#....#.."
            "..#...##.."
            "...###.#.."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'b',
            ".........."
            ".........."
            "..#......."
            "..#......."
            "..#####..."
            "..#....#.."
            "..#....#.."
            "..#####..."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'c',
            ".........."
            ".........."
            ".........."
            "...####..."
            "..#....#.."
            "..#......."
            "..#....#.."
            "...####..."

            ".........."
            ".........."
            );
    theFont_10x8->AddChar
        (
            'd',
            ".........."
            ".........."
            ".......#.."
            ".......#.."
            "...#####.."
            "..#....#.."
            "..#....#.."
            "...#####.."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'e',
            ".........."
            ".........."
            "...####..."
            "..#....#.."
            "..#....#.."
            "..#####..."
            "..#......."
            "...####..."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'f',
            ".........."
            ".....###.."
            "....#....."
            "...#......"
            "..####...."
            "...#......"
            "...#......"
            "...#......"

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'g',
            ".........."
            ".........."
            ".........."
            "...###.#.."
            "..#...##.."
            "..#....#.."
            "..#....#.."
            "...#####.."

            ".......#.."
            "...####..."
            );

    theFont_10x8->AddChar
        (
            'h',
            ".........."
            "..#......."
            "..#......."
            "..#......."
            "..#####..."
            "..#....#.."
            "..#....#.."
            "..#....#.."

            ".........."
            ".........."
            );





    theFont_10x8->AddChar
        (
            'i',
            ".........."
            "...#......"
            ".........."
            "..##......"
            "...#......"
            "...#......"
            "....#....."
            ".....##..."

            ".........."
            ".........."
            );
    theFont_10x8->AddChar
        (
            'j',
            ".........."
            "......#..."
            ".........."
            "......#..."
            "......#..."
            "......#..."
            "......#..."
            "......#..."

            ".....#...."
            "..###....."
            );
    theFont_10x8->AddChar
        (
            'k',
            ".........."
            "..#......."
            "..#......."
            "..#...#..."
            "..#..#...."
            "..###....."
            "..#..#...."
            "..#...#..."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'l',
            "..##......"
            "...#......"
            "...#......"
            "...#......"
            "...#......"
            "...#......"
            "....#..#.."
            ".....##..."

            ".........."
            ".........."
            );


    theFont_10x8->AddChar
        (
            'm',
            ".........."
            ".........."
            "..#.#.##.."
            "..##.#..#."
            "..#..#..#."
            "..#..#..#."
            "..#..#..#."
            "..#..#..#."

            ".........."
            ".........."
            );
    theFont_10x8->AddChar
        (
            'n',
            ".........."
            ".........."
            "..#.####.."
            "..##....#."
            "..#.....#."
            "..#.....#."
            "..#.....#."
            "..#.....#."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'o',
            ".........."
            ".........."
            "....###..."
            "...#...#.."
            "..#.....#."
            "..#.....#."
            "...#...#.."
            "....###..."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'p',
            ".........."
            ".........."
            "..####...."
            "..#...#..."
            "..#....#.."
            "..#....#.."
            "..#...#..."
            "..####...."
            "..#......."
            "..#......."
            );


    theFont_10x8->AddChar
        (
            'q',
            ".........."
            ".........."
            "...###.#.."
            "..#...##.."
            "..#....#.."
            "..#....#.."
            "..#...##.."
            "...###.#.."
            ".......#.."
            ".......#.."
            );

    theFont_10x8->AddChar
        (
            'r',
            ".........."
            ".........."
            "..#.####.."
            "..##......"
            "..#......."
            "..#......."
            "..#......."
            "..#......."

            ".........."
            ".........."
            );


    theFont_10x8->AddChar
        (
            's',
            ".........."
            ".........."
            "...####..."
            "..#......."
            "...####..."
            ".......#.."
            "..#....#.."
            "...####..."

            ".........."
            ".........."
            );


    theFont_10x8->AddChar
        (
            't',
            ".........."
            ".........."
            "...#......"
            "..####...."
            "...#......"
            "...#......"
            "...#...#.."
            "....###..."

            ".........."
            ".........."
            );
    theFont_10x8->AddChar
        (
            'u',
            ".........."
            ".........."
            "..#.....#."
            "..#.....#."
            "..#.....#."
            "..#.....#."
            "..#....##."
            "...####.#."

            ".........."
            ".........."
            );
    theFont_10x8->AddChar
        (
            'v',
            ".........."
            ".........."
            "..#.....#."
            "..#.....#."
            "...#...#.."
            "...#...#.."
            "....#.#..."
            ".....#...."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'w',
            ".........."
            ".........."
            "#.......#."
            "#.......#."
            ".#..#..#.."
            ".#..#..#.."
            "..##.##..."
            "..#...#..."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'x',
            ".........."
            ".........."
            ".........."
            "..#....#.."
            "...#..#..."
            "....##...."
            "...#..#..."
            "..#....#.."

            ".........."
            ".........."
            );

    theFont_10x8->AddChar
        (
            'y',
            ".........."
            ".........."
            ".........."
            "..#.....#."
            "...#...#.."
            "....###..."
            ".....#...."
            ".....#...."

            ".........."
            ".........."
            );


    theFont_10x8->AddChar
        (
            'z',
            ".........."
            ".........."
            "..######.."
            "......#..."
            ".....#...."
            "....#....."
            "...#......"
            "..######.."

            ".........."
            ".........."
            );

    return *theFont_10x8;
}

} // namespace Private
