#ifndef CBLENDINGORTHOS_H
#define CBLENDINGORTHOS_H
#include "cero_modelonepaire.h"

class cAppli_BlendingOrthos
{
public:
    cAppli_BlendingOrthos(int argc, char** argv);

private:
    cInterfChantierNameManipulateur * mICNM;
    std::string mFullName;
    std::string mDir;
    std::string mPat;
    std::list<std::string> mLFile;
    std::vector<cBlendingOrthos *>           mVOs; // vecteur d'orthos

};

// classe ortho contenant 1) la radiométrie 2) la zone ou on effectue le blending

class cBlendingOrthos
{
public:
    cBlendingOrthos(cAppli_BlendingOrthos & anAppli,const std::string & aName);
    Im2D_INT1 * AreaInMosaic(){return &mAreaInMosaic;}
    Im2D_INT1 * blendingArea(){return &mBlendingArea;}
    Im2D_REAL4 * rad(){return &mIm;} // radiometry
    // receive num of label and label map and feathering distance in pixels, distance d5711 op morpho
    void SetAreaInMosaicAndBlendingZ(const mLabelMap);

private:
    Im2D_INT1   mAreaInMosaic;
    Im2D_INT1   mBlendingArea;
    Im2D_REAL4  mIm;
    std::string mName;
    int         mInd;
    cAppli_BlendingOrthos & mAppli;

};

#endif // CBLENDINGORTHOS_H
