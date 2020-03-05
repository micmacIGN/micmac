#include "Sat.h"
#include "../uti_phgrm/MICMAC/cCameraModuleOrientation.h"
#include "../uti_phgrm/MICMAC/Jp2ImageLoader.h"
#include "../uti_phgrm/MICMAC/cInterfModuleImageLoader.h"

#include "Surf.h"

#ifdef WIN32
#else
    #ifndef __APPLE__
        #pragma GCC diagnostic push
    #endif
    #pragma GCC diagnostic ignored "-Wunused-result"
#endif


std::string baseName(std::string const &aName)
{
    //on recupere l'extension
    int placePoint = -1;
    for(int l=(int)(aName.size()-1);(l>=0)&&(placePoint==-1);--l)
    {
        if (aName[l]=='.')
        {
            placePoint = l;
        }
    }
    std::string base = std::string("");
    if (placePoint!=-1)
    {
        base.assign(aName.begin(),aName.begin()+placePoint);
    }
    else
    {
        base = aName;
    }
    return base;
}

Pt2di getImageSize(std::string const &aName)
{
    //on recupere l'extension
    int placePoint = -1;
    for(int l=(int)(aName.size()-1);(l>=0)&&(placePoint==-1);--l)
    {
        if (aName[l]=='.')
        {
            placePoint = l;
        }
    }
    std::string ext = std::string("");
    if (placePoint!=-1)
    {
        ext.assign(aName.begin()+placePoint+1,aName.end());
    }
    //std::cout << "Extension : "<<ext<<std::endl;

#if defined (__USE_JP2__)
    // on teste l'extension
    if ((ext==std::string("jp2"))|| (ext==std::string("JP2")) || (ext==std::string("Jp2")))
    {
        //std::cout<<"JP2 avec Jp2ImageLoader"<<std::endl;
        std_unique_ptr<cInterfModuleImageLoader> aRes(new JP2ImageLoader(aName, false));
        if (aRes.get())
        {
            return Std2Elise(aRes->Sz(1));
        }
    }
#endif

    Tiff_Im aTif = Tiff_Im::StdConvGen(aName,1,true,false);
    return aTif.sz();
}

void getImageInfo(std::string const &aName,Pt2di &Size,int &NbCanaux)
{
    Size.x = 0;
    Size.y = 0;
    NbCanaux = 0;
    //on recupere l'extension
    int placePoint = -1;
    for(int l=(int)(aName.size()-1);(l>=0)&&(placePoint==-1);--l)
    {
        if (aName[l]=='.')
        {
            placePoint = l;
        }
    }
    std::string ext = std::string("");
    if (placePoint!=-1)
    {
        ext.assign(aName.begin()+placePoint+1,aName.end());
    }
    //std::cout << "Extension : "<<ext<<std::endl;
    
#if defined (__USE_JP2__)
    // on teste l'extension
    if ((ext==std::string("jp2"))|| (ext==std::string("JP2")) || (ext==std::string("Jp2")))
    {
        //std::cout<<"JP2 avec Jp2ImageLoader"<<std::endl;
        std_unique_ptr<cInterfModuleImageLoader> aRes(new JP2ImageLoader(aName, false));
        if (aRes.get())
        {
            Size = Std2Elise(aRes->Sz(1));
            NbCanaux = aRes->NbCanaux();
            return;
        }
    }
#endif
    
    Tiff_Im aTif = Tiff_Im::StdConvGen(aName,1,true,false);
    Size = aTif.sz();

}


template <class Type,class TyBase>
std::vector<TIm2D<Type,TyBase>* > createVTIm2DFromFile(std::string const &aName, Pt2di const &PminCrop,Pt2di const &SzCrop)
{
    std::vector<TIm2D<Type,TyBase>* > vPtr;
    //on recupere l'extension
    int placePoint = -1;
    for(int l=(int)(aName.size()-1);(l>=0)&&(placePoint==-1);--l)
    {
        if (aName[l]=='.')
        {
            placePoint = l;
        }
    }
    std::string ext = std::string("");
    if (placePoint!=-1)
    {
        ext.assign(aName.begin()+placePoint+1,aName.end());
    }
    //std::cout << "Extension : "<<ext<<std::endl;
    
#if defined (__USE_JP2__)
    // on teste l'extension
    if ((ext==std::string("jp2"))|| (ext==std::string("JP2")) || (ext==std::string("Jp2")))
    {
        //std::cout<<"JP2 avec Jp2ImageLoader"<<std::endl;
        std_unique_ptr<cInterfModuleImageLoader> aRes(new JP2ImageLoader(aName, false));
        if (aRes.get()!=NULLPTR)
        {
            int mFlagLoadedIms=0;
            std::vector<sLowLevelIm<Type> > vSLLI;

            for(int i=0;i<aRes->NbCanaux();++i)
            {
                TIm2D<Type,TyBase> * ptr = new TIm2D<Type,TyBase>(SzCrop);
                vPtr.push_back(ptr);
                vSLLI.push_back(sLowLevelIm<Type>
                               (
                                ptr->_the_im.data_lin(),
                                ptr->_the_im.data(),
                                Elise2Std(ptr->sz())
                                ));
                mFlagLoadedIms += (1<<i);
            }
            aRes->LoadNCanaux(vSLLI,mFlagLoadedIms,
                                  1,//deZoom
                                  cInterfModuleImageLoader::tPInt(0,0),//aP0Im
                                  cInterfModuleImageLoader::tPInt(PminCrop.x,PminCrop.y),//aP0File
                                  cInterfModuleImageLoader::tPInt(SzCrop.x,SzCrop.y));
        }
    }
#endif
    
//    std_unique_ptr<TIm2D<Type,TyBase> > anTIm2D(new TIm2D<Type,TyBase>(SzCrop));
//    Tiff_Im aTif = Tiff_Im::StdConvGen(aName,1,true,false);
//    ELISE_COPY(anTIm2D->_the_im.all_pts(),trans(aTif.in(),PminCrop),anTIm2D->_the_im.out());
    return vPtr;
}


template <class Type,class TyBase>
TIm2D<Type,TyBase>* createTIm2DFromFile(std::string const &aName, Pt2di const &PminCrop,Pt2di const &SzCrop)
{
    //on recupere l'extension
    int placePoint = -1;
    for(int l=(int)(aName.size()-1);(l>=0)&&(placePoint==-1);--l)
    {
        if (aName[l]=='.')
        {
            placePoint = l;
        }
    }
    std::string ext = std::string("");
    if (placePoint!=-1)
    {
        ext.assign(aName.begin()+placePoint+1,aName.end());
    }
    //std::cout << "Extension : "<<ext<<std::endl;

#if defined (__USE_JP2__)
    // on teste l'extension
    if ((ext==std::string("jp2"))|| (ext==std::string("JP2")) || (ext==std::string("Jp2")))
    {
        //std::cout<<"JP2 avec Jp2ImageLoader"<<std::endl;
        std_unique_ptr<cInterfModuleImageLoader> aRes(new JP2ImageLoader(aName, false));
        if (aRes.get()!=NULLPTR)
        {
            std_unique_ptr<TIm2D<Type,TyBase> > anTIm2D(new TIm2D<Type,TyBase>(SzCrop));
            aRes->LoadCanalCorrel(sLowLevelIm<Type>
                                  (
                                   anTIm2D->_the_im.data_lin(),
                                   anTIm2D->_the_im.data(),
                                   Elise2Std(anTIm2D->sz())
                                   ),
                                  1,//deZoom
                                  cInterfModuleImageLoader::tPInt(0,0),//aP0Im
                                  cInterfModuleImageLoader::tPInt(PminCrop.x,PminCrop.y),//aP0File
                                  cInterfModuleImageLoader::tPInt(SzCrop.x,SzCrop.y));
            return anTIm2D.release();
        }
    }
#endif

    std_unique_ptr<TIm2D<Type,TyBase> > anTIm2D(new TIm2D<Type,TyBase>(SzCrop));
    Tiff_Im aTif = Tiff_Im::StdConvGen(aName,1,true,false);
    ELISE_COPY(anTIm2D->_the_im.all_pts(),trans(aTif.in(),PminCrop),anTIm2D->_the_im.out());
    return anTIm2D.release();
}


template <class Type,class TyBase>
std::vector<TIm2D<Type,TyBase>* > createVTIm2DFromFile(std::string const &aName)
{
    std::vector<TIm2D<Type,TyBase>* > vPtr;
    //on recupere l'extension
    int placePoint = -1;
    for(int l=aName.size()-1;(l>=0)&&(placePoint==-1);--l)
    {
        if (aName[l]=='.')
        {
            placePoint = l;
        }
    }
    std::string ext = std::string("");
    if (placePoint!=-1)
    {
        ext.assign(aName.begin()+placePoint+1,aName.end());
    }
    //std::cout << "Extension : "<<ext<<std::endl;
    
#if defined (__USE_JP2__)
    // on teste l'extension
    if ((ext==std::string("jp2"))|| (ext==std::string("JP2")) || (ext==std::string("Jp2")))
    {
        std_unique_ptr<cInterfModuleImageLoader> aRes(new JP2ImageLoader(aName, false));
        if (aRes.get()!=NULLPTR)
        {
            Pt2di aSz = Std2Elise(aRes->Sz(1));
            
            
            int mFlagLoadedIms=0;
            std::vector<sLowLevelIm<Type> > vSLLI;
            
            for(size_t i=0;i<aRes->NbCanaux();++i)
            {
                TIm2D<Type,TyBase> * ptr = new TIm2D<Type,TyBase>(aSz);
                vPtr.push_back(ptr);
                vSLLI.push_back(sLowLevelIm<Type>
                               (
                                ptr->_the_im.data_lin(),
                                ptr->_the_im.data(),
                                Elise2Std(ptr->sz())
                                ));
                mFlagLoadedIms += (1<<i);
            }
            
            aRes->LoadNCanaux(vSLLI,mFlagLoadedIms,
                                  1,//deZoom
                                  cInterfModuleImageLoader::tPInt(0,0),//aP0Im
                                  cInterfModuleImageLoader::tPInt(0,0),//aP0File
                                  cInterfModuleImageLoader::tPInt(aSz.x,aSz.y));
        }
    }
#endif
    return vPtr;
    // Attention la version FromFileStd ne fonctionne pas avec une image couleur??
    //return new TIm2D<Type,TyBase>(Im2D<Type,TyBase>::FromFileStd(aName));
    //return new TIm2D<Type,TyBase>(Im2D<Type,TyBase>::FromFileBasic(aName));
}

template <class Type,class TyBase>
TIm2D<Type,TyBase>* createTIm2DFromFile(std::string const &aName)
{
    //on recupere l'extension
    int placePoint = -1;
    for(int l=(int)(aName.size()-1);(l>=0)&&(placePoint==-1);--l)
    {
        if (aName[l]=='.')
        {
            placePoint = l;
        }
    }
    std::string ext = std::string("");
    if (placePoint!=-1)
    {
        ext.assign(aName.begin()+placePoint+1,aName.end());
    }
    //std::cout << "Extension : "<<ext<<std::endl;

#if defined (__USE_JP2__)
    // on teste l'extension
    if ((ext==std::string("jp2"))|| (ext==std::string("JP2")) || (ext==std::string("Jp2")))
    {
        std_unique_ptr<cInterfModuleImageLoader> aRes(new JP2ImageLoader(aName, false));
        if (aRes.get()!=NULLPTR)
        {
            Pt2di aSz = Std2Elise(aRes->Sz(1));
            std_unique_ptr<TIm2D<Type,TyBase> > anTIm2D(new TIm2D<Type,TyBase>(aSz));
            aRes->LoadCanalCorrel(sLowLevelIm<Type>
                                   (
                                    anTIm2D->_the_im.data_lin(),
                                    anTIm2D->_the_im.data(),
                                    Elise2Std(anTIm2D->sz())
                                    ),
                                   1,
                                   cInterfModuleImageLoader::tPInt(0,0),
                                   cInterfModuleImageLoader::tPInt(0,0),
                                   cInterfModuleImageLoader::tPInt(aSz.x,aSz.y));
            return anTIm2D.release();
        }
    }
#endif
    // Attention la version FromFileStd ne fonctionne pas avec une image couleur??
    //return new TIm2D<Type,TyBase>(Im2D<Type,TyBase>::FromFileStd(aName));
    return new TIm2D<Type,TyBase>(Im2D<Type,TyBase>::FromFileBasic(aName));
}

double correl(std::vector<double> const &f1,std::vector<double> const &f2)
{
    std::vector<double>::const_iterator it1,it2,fin1,fin2;
    it1=f1.begin();
    it2=f2.begin();
    fin1=f1.end();
    fin2=f2.end();

    double sa=0.,sa2=0.,sb=0.,sb2=0,sab=0.;
    int N=0;
    for(;(it1!=fin1)&&(it2!=fin2);)
    {
        sa += (*it1);
        sa2 += pow((*it1),2);
        sb += (*it2);
        sb2 += pow((*it2),2);
        sab += (*it1)*(*it2);
        ++N;
        ++it1;
        ++it2;
    }
    double pdtVar = (sa2 - pow(sa,2)/N)*(sb2-pow(sb,2)/N);
    if (pdtVar!=0.)
        return (sab-sa*sb/N)/sqrt(pdtVar);
    return 0.;
}

double correl(std::vector<double> const &f1,std::vector<double> const &f2,double NoData)
{
    std::vector<double>::const_iterator it1,it2,fin1,fin2;
    it1=f1.begin();
    it2=f2.begin();
    fin1=f1.end();
    fin2=f2.end();

    double sa=0.,sa2=0.,sb=0.,sb2=0,sab=0.;
    int N=0;
    for(;(it1!=fin1)&&(it2!=fin2);)
    {
        if (((*it1)!=NoData)&&((*it2)!=NoData))
        {
            sa += (*it1);
            sa2 += pow((*it1),2);
            sb += (*it2);
            sb2 += pow((*it2),2);
            sab += (*it1)*(*it2);
            ++N;
        }
        ++it1;
        ++it2;
    }
    if (N == 0)
        return 0.;
    double pdtVar = (sa2 - pow(sa,2)/N)*(sb2-pow(sb,2)/N);
    if (pdtVar==0.)
        return 0.;
    return (sab-sa*sb/N)/sqrt(pdtVar);
}


double dist(DigeoPoint const &pt1, DigeoPoint const &pt2)
{
    double dmin=0.;
    bool first = true;

    for (size_t i=0;i<pt1.entries.size();++i)
    {
        DigeoPoint::Entry const &entrie1 = pt1.entries[i];
        for (size_t j=0;j<pt2.entries.size();++j)
        {
            DigeoPoint::Entry const &entrie2 = pt2.entries[j];
            double d = 0.;
            for( int k=0;k<DIGEO_DESCRIPTOR_SIZE;++k)
            {
                d += pow(entrie1.descriptor[k] - entrie2.descriptor[k], 2);
            }
            if (first)
            {
                dmin = d;
            }
            else
            {
                first = false;
                if (dmin>d)
                {
                    dmin = d;
                }
            }
        }
    }
    return dmin;
}

int TP2GCP(std::string const &aNameFileMNT,
               std::string const &aNameFileOrtho,
               std::string const &aNameFilePOIOrtho,
               std::string const &aNameFileGrid,
               std::string const &aNameFileImage,
               std::string const &aNameFilePOIImage,
               double seuilPixel,
               double seuilCorrel,
               int SzW,
               std::string const &aNameResult)
{
    int SzMaxImg = 4000 * 4000;

    // Chargement du MNT
    cFileOriMnt aMntOri=  StdGetFromPCP(aNameFileMNT,FileOriMnt);
    std::cout << "Taille du MNT : "<<aMntOri.NombrePixels().x<<" "<<aMntOri.NombrePixels().y<<std::endl;
    std_unique_ptr<TIm2D<REAL4,REAL8> > aMntImg(createTIm2DFromFile<REAL4,REAL8>(aMntOri.NameFileMnt()));
    if (aMntImg.get()==NULLPTR)
    {
        cerr << "Error in "<<aMntOri.NameFileMnt()<<std::endl;
        return EXIT_FAILURE;
    }

    // Chargement de l'Ortho
    cFileOriMnt aOrthoOri=  StdGetFromPCP(aNameFileOrtho,FileOriMnt);
    std::cout << "Taille de l'ortho : "<<aOrthoOri.NombrePixels().x<<" "<<aOrthoOri.NombrePixels().y<<std::endl;
    std_unique_ptr<TIm2D<U_INT1,INT4> > OrthoImg(NULLPTR);
    Pt2di OrthoSz = getImageSize(aOrthoOri.NameFileMnt());
    if (OrthoSz.x * OrthoSz.y < SzMaxImg)
    {
        OrthoImg.reset(createTIm2DFromFile<U_INT1,INT4>(aOrthoOri.NameFileMnt()));
        if (OrthoImg.get()==NULL)
        {
            cerr << "Error in "<<aOrthoOri.NameFileMnt()<<std::endl;
            return EXIT_FAILURE;
        }
    }

    // Chargement des points d'interet dans l'ortho
    vector<DigeoPoint> keypointsOrtho;

    if ( !DigeoPoint::readDigeoFile( aNameFilePOIOrtho, true, keypointsOrtho ) ){
        cerr << "WARNING: unable to read keypoints in [" << aNameFilePOIOrtho << "]" << endl;
        return EXIT_FAILURE;
    }
    std::cout << "Nombre de points dans l'ortho : "<<keypointsOrtho.size()<<std::endl;

    // Chargement des points d'interet dans l'image
    vector<DigeoPoint> keypointsImage;

    if ( !DigeoPoint::readDigeoFile( aNameFilePOIImage, true, keypointsImage ) ){
        cerr << "WARNING: unable to read keypoints in [" << aNameFilePOIImage << "]" << endl;
        return EXIT_FAILURE;
    }
    std::cout << "Nombre de points dans l'image : "<<keypointsImage.size()<<std::endl;

    // Chargement de la grille et de l'image
    std_unique_ptr<TIm2D<U_INT1,INT4> > img(NULLPTR);
    Pt2di ImgSz = getImageSize(aNameFileImage);
    if (ImgSz.x * ImgSz.y < SzMaxImg)
    {
        img.reset(createTIm2DFromFile<U_INT1,INT4>(aNameFileImage));
        if (img.get()==NULL)
        {
            cerr << "Error in "<<aNameFileImage<<std::endl;
            return EXIT_FAILURE;
        }
    }

    // Chargement de la grille et de l'image
    ElAffin2D oriIntImaM2C;
    std_unique_ptr<ElCamera> aCamera(new cCameraModuleOrientation(new OrientationGrille(aNameFileGrid),ImgSz,oriIntImaM2C));

    bool verbose=true;

    vector<DigeoPoint>::const_iterator it,fin=keypointsOrtho.end();
    vector<DigeoPoint>::const_iterator it2,fin2=keypointsImage.end();

    double seuilPixel2 = pow(seuilPixel,2);

    std::ofstream fic(aNameResult.c_str());
    fic << std::setprecision(15);

    // Export sous forme de dico appuis et mesures
    cSetOfMesureAppuisFlottants aSetDicoMesure;

    cMesureAppuiFlottant1Im aDicoMesure;
    aDicoMesure.NameIm()=aNameFileImage;
    cDicoAppuisFlottant aDicoAppuis;
    int cpt = 0;

    if (verbose) std::cout << "aOrthoOri.ResolutionPlani() : "<<aOrthoOri.ResolutionPlani()<<" / aOrthoOri.OriginePlani() : "<<aOrthoOri.OriginePlani()<<std::endl;

    double NoData = -9999.;

    // on parcourt les points sift de l'ortho
    for(it=keypointsOrtho.begin();it!=fin;++it)
    {
        DigeoPoint const &ptSift = (*it);
        // On estime la position 3D du point
        // ToDo: ajouter l'interpolation de l'alti dans le mnt
        Pt2dr ptOrtho(ptSift.x,ptSift.y);
        //if (verbose) std::cout << "Point img dans l'ortho : "<<ptOrtho.x<<" "<<ptOrtho.y<<std::endl;
        ptOrtho.x =  ptOrtho.x*aOrthoOri.ResolutionPlani().x + aOrthoOri.OriginePlani().x;
        ptOrtho.y =  ptOrtho.y*aOrthoOri.ResolutionPlani().y + aOrthoOri.OriginePlani().y;
        //if (verbose) std::cout << "Point terrain 2D : "<<ptOrtho.x<<" "<<ptOrtho.y<<std::endl;
        // Position dans le MNT
        Pt2dr ptMnt;
        ptMnt.x = (ptOrtho.x-aMntOri.OriginePlani().x)/aMntOri.ResolutionPlani().x;
        ptMnt.y = (ptOrtho.y-aMntOri.OriginePlani().y)/aMntOri.ResolutionPlani().y;
        //if (verbose) std::cout << "Point img dans le Mnt : "<<ptMnt.x<<" "<<ptMnt.y<<std::endl;
        double alti = aMntImg->getr(ptMnt,NoData)*aMntOri.ResolutionAlti() + aMntOri.OrigineAlti();
        if (alti == NoData)
        {
            std::cout << "Pas d'altitude trouvee pour le point : "<<ptOrtho.x<<" "<<ptOrtho.y<<std::endl;
            std::cout << "On passe au point suivant"<<std::endl;
            break;
        }
        //if (verbose) std::cout << "Altitude Mnt : "<<alti<<std::endl;
        // Position dans l'image
        Pt3dr pt3(ptOrtho.x,ptOrtho.y,alti);
        //if (verbose) std::cout << "Point terrain : "<<pt3.x<<" "<<pt3.y<<" "<<pt3.z<<std::endl;
        Pt2dr pImg = aCamera->R3toF2(pt3);
        //if (verbose) std::cout << "Point Image : "<<pImg.x<<" "<<pImg.y<<std::endl;


        std::vector<double> fenOrtho;
        for(int l=ptSift.y-SzW;l<=(ptSift.y+SzW);++l)
        {
            for(int c=ptSift.x-SzW;c<=(ptSift.x+SzW);++c)
            {
                fenOrtho.push_back(OrthoImg->getr(Pt2dr(c,l),NoData));
            }
        }


        DigeoPoint ptHomoDmin;
        Pt3dr pTerrainDmin;
        double dmin = -1.;
        // On cherche un POIImage proche de pImg
        for(it2=keypointsImage.begin();it2!=fin2;++it2)
        {
            DigeoPoint const &ptSift2 = (*it2);
            Pt3dr pTerrain = aCamera->F2AndZtoR3(Pt2dr(ptSift2.x,ptSift2.y),pt3.z);

            double d2 = pow(pImg.x-ptSift2.x,2) + pow(pImg.y-ptSift2.y,2);
            if (d2<seuilPixel2)
            {
                double d = dist(ptSift,ptSift2);
                if ((dmin<0)||(dmin>d))
                {
                    dmin = d;
                    ptHomoDmin = ptSift2;
                    pTerrainDmin = pTerrain;
                }
            }
        }
        if (dmin>=0)
        {
            // On valide le point avec la correlation

            std::vector<double> fenImage;

            if (img.get()!=NULL)
            {
                // version sans crop
                for(int dy=-SzW;dy<=SzW;++dy)
                {
                    for(int dx=-SzW;dx<=SzW;++dx)
                    {
                        Pt3dr P3D;
                        P3D.x = pTerrainDmin.x + dx * aOrthoOri.ResolutionPlani().x;
                        P3D.y = pTerrainDmin.y + dy * aOrthoOri.ResolutionPlani().y;
                        P3D.z = pTerrainDmin.z;

                        Pt2dr p2 = aCamera->R3toF2(P3D);
                        fenImage.push_back(img->getr(p2,NoData));
                    }
                }
            }
            else
            {
                // version avec crop
                std::vector<Pt2dr> vCoordImage;
                double cmin = 0 ,cmax = 0 ,lmin = 0,lmax = 0;
                for(int dy=-SzW;dy<=SzW;++dy)
                {
                    for(int dx=-SzW;dx<=SzW;++dx)
                    {
                        Pt3dr P3D;
                        P3D.x = pTerrainDmin.x + dx * aOrthoOri.ResolutionPlani().x;
                        P3D.y = pTerrainDmin.y + dy * aOrthoOri.ResolutionPlani().y;
                        P3D.z = pTerrainDmin.z;

                        Pt2dr p2 = aCamera->R3toF2(P3D);
                        vCoordImage.push_back(p2);
                        if (vCoordImage.size()==1)
                        {
                            cmin = p2.x;
                            cmax = p2.x;
                            lmin = p2.y;
                            lmax = p2.y;
                        }
                        else
                        {
                            if (cmin>p2.x)
                                cmin=p2.x;
                            else if (cmax<p2.x)
                                cmax=p2.x;
                            if (lmin>p2.y)
                                lmin=p2.y;
                            else if (lmax<p2.y)
                                lmax=p2.y;
                        }
                    }
                }
                Pt2di PminCrop((int)round_ni(cmin-1.),(int)round_ni(lmin-1.));
                Pt2di SzCrop((int)round_ni(cmax-PminCrop.x+2),(int)round_ni(lmax-PminCrop.y+2));
                //std::cout << "Crop : "<<PminCrop.x<<" "<<PminCrop.y<<" / "<<SzCrop.x<<" "<<SzCrop.y<<std::endl;
                std_unique_ptr<TIm2D<U_INT1,INT4> > cropImg(createTIm2DFromFile<U_INT1,INT4>(aNameFileImage,PminCrop,SzCrop));
                if (cropImg.get()==NULL)
                {
                    cerr << "Error in "<<aNameFileImage<<" Crop : "<<PminCrop.x<<" "<<PminCrop.y<<" / "<<SzCrop.x<<" "<<SzCrop.y<<std::endl;
                    return EXIT_FAILURE;
                }
                for(size_t i=0;i<vCoordImage.size();++i)
                {
                    fenImage.push_back(cropImg->getr(Pt2dr(vCoordImage[i].x-PminCrop.x,vCoordImage[i].y-PminCrop.y),NoData));
                }
            }

            // Coeff de correlation
            double coef = correl(fenOrtho,fenImage,NoData);
            if (coef >= seuilCorrel)
            {
                if (verbose) std::cout << "Point Dmin : "<<ptHomoDmin.x<<" "<<ptHomoDmin.y<<" distSift="<<dmin<<" Correl="<<coef<<std::endl;
                fic << ptSift.x<<" "<<ptSift.y<<" "<<ptHomoDmin.x<<" "<<ptHomoDmin.y<<std::endl;

                cOneMesureAF1I aMesure;
                cOneAppuisDAF aAppui;

                Pt2dr pt(ptSift.x,ptSift.y);
                aMesure.PtIm()=pt;

                aAppui.Pt() = pt3;
                aAppui.Incertitude()=Pt3dr(1,1,1);

                std::ostringstream oss;
                oss << "Point_"<<cpt;
                ++cpt;
                aAppui.NamePt()=oss.str();
                aMesure.NamePt()=oss.str();

                aDicoAppuis.OneAppuisDAF().push_back(aAppui);
                aDicoMesure.OneMesureAF1I().push_back(aMesure);
            }
        }

        // Autre approche: on teste tous le voisinage
/*
        double coefmax = -1.;
        Pt2dr ptMaxCorrel;
        for(int dl = -seuilPixel;dl<=seuilPixel;++dl)
        {
            for(int dc = -seuilPixel;dc<=seuilPixel;++dc)
            {
                Pt2dr pt2;
                pt2.x = pImg.x + dc;
                pt2.y = pImg.y + dl;
                Pt3dr pTerrain = aCamera->F2AndZtoR3(Pt2dr(pt2.x,pt2.y),pt3.z);
                std::vector<double> fenImage;
                // Plutot que de reprojetter tous les points de la fenetre, on interpole la projection
                Pt3dr pTerrain2;
                pTerrain2.x = pTerrain.x + aOrthoOri.ResolutionPlani().x;
                pTerrain2.y = pTerrain.y;
                pTerrain2.z = pTerrain.z;
                Pt2dr ptImg2 = aCamera->R3toF2(pTerrain2);
                double deltax_col = ptImg2.x - pt2.x;
                double deltax_lig = ptImg2.y - pt2.y;

                pTerrain2.x = pTerrain.x;
                pTerrain2.y = pTerrain.y + aOrthoOri.ResolutionPlani().y;
                pTerrain2.z = pTerrain.z;
                ptImg2 = aCamera->R3toF2(pTerrain2);
                double deltay_col = ptImg2.x - pt2.x;
                double deltay_lig = ptImg2.y - pt2.y;

                for(int dy=-SzW;dy<=SzW;++dy)
                {
                    for(int dx=-SzW;dx<=SzW;++dx)
                    {
                        Pt2dr p2;
                        p2.x = pt2.x + dx * deltax_col + dy * deltay_col;
                        p2.y = pt2.y + dx * deltax_lig + dy * deltay_lig;
                        fenImage.push_back(img.getr(p2,NoData));
                    }
                }
                // Coeff de correlation
                double coef = correl(fenOrtho,fenImage,NoData);
                //std::cout << "Correl : "<<coef<<std::endl;
                if (coef>coefmax)
                {
                    coefmax = coef;
                    ptMaxCorrel = pt2;
                }
            }
        }
        if (verbose) std::cout << "Max de correlation : "<<ptMaxCorrel.x<<" "<<ptMaxCorrel.y<<" correl = "<<coefmax<<std::endl;
        if (coefmax>=seuilCorrel)
            fic2 << ptSift.x<<" "<<ptSift.y<<" "<<ptMaxCorrel.x<<" "<<ptMaxCorrel.y<<" "<<coefmax<<std::endl;
*/
    }

    //on recupere le nom de sortie sans extension
    int placePoint = -1;
    for(int l=(int)(aNameResult.size()-1);(l>=0)&&(placePoint==-1);--l)
    {
        if (aNameResult[l]=='.')
        {
            placePoint = l;
        }
    }
    std::string aBaseNameResult = aNameResult;
    if (placePoint!=-1)
    {
        aBaseNameResult.assign(aNameResult.begin(),aNameResult.begin()+placePoint);
    }
    std::cout << "Nom de sortie : "<<aBaseNameResult<<std::endl;


     aSetDicoMesure.MesureAppuiFlottant1Im().push_back(aDicoMesure);
     MakeFileXML(aDicoAppuis,aBaseNameResult+"-S3D.xml");
     MakeFileXML(aSetDicoMesure,aBaseNameResult+"-S2D.xml");

    return EXIT_SUCCESS;
}


Pt3dr Img2Terrain(ElCamera *aCamera, TIm2D<REAL4,REAL8> *mnt, cFileOriMnt const &ori,
                  double Zinit,
                  Pt2di const &Pimg)
{
    bool verbose = false;
    int itMax = 10;
    double seuilZ = 1.;
    double NoData = -9999.;


    double Z = Zinit;
    Pt3dr Pterr, Pterr2;
    Pt2dr Pmnt;

    int it = 0;
    bool fin = false;
    while((it<itMax)&&!fin)
    {
        Pterr = aCamera->F2AndZtoR3(Pt2dr(Pimg.x,Pimg.y),Z);
        if (verbose) std::cout << "it: "<<it<<" Pterr: "<<Pterr.x<<" "<<Pterr.y<<" "<<Pterr.z<<std::endl;
        // Position dans le MNT
        Pmnt.x = (Pterr.x-ori.OriginePlani().x)/ori.ResolutionPlani().x;
        Pmnt.y = (Pterr.y-ori.OriginePlani().y)/ori.ResolutionPlani().y;
        if (verbose) std::cout << "Pmnt : "<<Pmnt<<std::endl;
        double vAlti = mnt->getr(Pmnt,NoData);
        if (vAlti == NoData)
        {
            std::cout << "Attention, le MNT ne couvre pas la zone : "<<Pmnt.x<<" "<<Pmnt.y<<std::endl;
        }
        double alti = vAlti*ori.ResolutionAlti() + ori.OrigineAlti();
        if (verbose) std::cout << "alti : "<<alti<<std::endl;
        fin = (std::abs(alti - Z)<=seuilZ);
        if (verbose) std::cout << "alti : "<<alti<<" Z : "<<Z<<" seuilZ "<<seuilZ<<" fin: "<<fin<<std::endl;
        Z = alti;
        ++it;
    }
    if (it == itMax)
        std::cout << "Attention, pas de convergence pour l'estimation des coordonnes 3D du point : "<<Pimg.x<<" "<<Pimg.y<<std::endl;

    return Pterr;
}

int Ortho(std::string const &aNameFileMNT,
          std::string const &aNameFileGrid,
          std::string const &aNameFileImage,
          double resolution,
          std::string const &aNameResult)
{
    int SzMaxImg = 10000 * 10000;
    
    // Chargement du MNT
    cFileOriMnt aMntOri=  StdGetFromPCP(aNameFileMNT,FileOriMnt);
    std::cout << "Taille du MNT : "<<aMntOri.NombrePixels().x<<" "<<aMntOri.NombrePixels().y<<std::endl;
    std::cout << "Chargement du fichier : "<<aMntOri.NameFileMnt()<<std::endl;
    std_unique_ptr<TIm2D<REAL4,REAL8> > aMntImg(createTIm2DFromFile<REAL4,REAL8>(aMntOri.NameFileMnt()));
    std::cout << "Fin du chargement"<<std::endl;
    if (aMntImg.get()==NULL)
    {
        cerr << "Error in "<<aMntOri.NameFileMnt()<<std::endl;
        return EXIT_FAILURE;
    }
    
    // Chargement de la grille et de l'image
    Pt2di ImgSz;
    int ImgNbC;
    getImageInfo(aNameFileImage,ImgSz,ImgNbC);

    
    std::cout << "Taille de l'image "<<aNameFileImage<<" : "<<ImgSz.x <<" "<<ImgSz.y<<" x "<<ImgNbC<<std::endl;
    
    std::vector<TIm2D<U_INT2,INT4>* > vBuffer;
    Pt2di bufferMin(0,0);
    Pt2di bufferMax(0,0);
    int tailleBuffer=8000;
    int margeBuffer=100;
    
    // Chargement de la grille et de l'image
    std::cout << "Chargement de la grille..."<<std::endl;
    ElAffin2D oriIntImaM2C;
    std_unique_ptr<ElCamera> aCamera(new cCameraModuleOrientation(new OrientationGrille(aNameFileGrid),ImgSz,oriIntImaM2C));
    std::cout << "...Fin"<<std::endl;
    
    //	bool verbose=true;
    
    // Recherche l'emprise de l'ortho a calculer
    
    double NoData = 0;
    double ZMoy = 0.;
    double xmin,ymin,xmax,ymax;
    // Projection des coins de l'image pour trouver l'emprise
    {
        Pt3dr Pterr = Img2Terrain(aCamera.get(),aMntImg.get(),aMntOri,ZMoy,Pt2di(0,0));
        xmin = Pterr.x;
        ymin = Pterr.y;
        xmax = Pterr.x;
        ymax = Pterr.y;
    }
    {
        Pt3dr Pterr = Img2Terrain(aCamera.get(),aMntImg.get(),aMntOri,ZMoy,Pt2di(ImgSz.x,0));
        if (xmin>Pterr.x)
            xmin = Pterr.x;
        else if (xmax<Pterr.x)
            xmax = Pterr.x;
        if (ymin>Pterr.y)
            ymin = Pterr.y;
        else if (ymax<Pterr.y)
            ymax = Pterr.y;
    }
    {
        Pt3dr Pterr = Img2Terrain(aCamera.get(),aMntImg.get(),aMntOri,ZMoy,Pt2di(ImgSz.x,ImgSz.y));
        if (xmin>Pterr.x)
            xmin = Pterr.x;
        else if (xmax<Pterr.x)
            xmax = Pterr.x;
        if (ymin>Pterr.y)
            ymin = Pterr.y;
        else if (ymax<Pterr.y)
            ymax = Pterr.y;
    }
    {
        Pt3dr Pterr = Img2Terrain(aCamera.get(),aMntImg.get(),aMntOri,ZMoy,Pt2di(0,ImgSz.y));
        if (xmin>Pterr.x)
            xmin = Pterr.x;
        else if (xmax<Pterr.x)
            xmax = Pterr.x;
        if (ymin>Pterr.y)
            ymin = Pterr.y;
        else if (ymax<Pterr.y)
            ymax = Pterr.y;
    }
    std::cout << "Emprise Terrain de l'image : "<<xmin<<" "<<ymin<<" "<<xmax<<" "<<ymax<<std::endl;
    Pt2dr P0Ortho(round_ni(xmin),round_ni(ymax));
    Pt2di SzOrtho((int)round_ni((xmax-xmin)/resolution),(int)round_ni((ymax-ymin)/resolution));
    std::cout << "Ortho : "<<P0Ortho.x<<" "<<P0Ortho.y<<" / "<<SzOrtho.x<<" "<<SzOrtho.y<<std::endl;
    
	if (( (unsigned int)SzOrtho.x*(unsigned int)SzOrtho.y )<(unsigned int)SzMaxImg)
    {
        std::cout << "Ortho n est pas trop grande, on peut la traiter en une fois"<<std::endl;
        // Creation de l'image
        std::vector<TIm2D<U_INT2,INT4>*> vPtrOrtho;
		for(size_t c=0;c<(unsigned int)ImgNbC;++c)
        {
            vPtrOrtho.push_back(new TIm2D<U_INT2,INT4>(SzOrtho));
        }
        
        
        // Remplissage de l'image
        std::cout << "Debut du remplissage de l ortho"<<std::endl;
        for(int l=0;l<SzOrtho.y;++l)
        {
            for(int c=0;c<SzOrtho.x;++c)
            {
                Pt2di Portho(c,l);
                Pt3dr Pterr;
                Pterr.x = P0Ortho.x + c * resolution;
                Pterr.y = P0Ortho.y - l * resolution;
                // Position dans le MNT
                Pt2dr Pmnt;
                Pmnt.x = (Pterr.x-aMntOri.OriginePlani().x)/aMntOri.ResolutionPlani().x;
                Pmnt.y = (Pterr.y-aMntOri.OriginePlani().y)/aMntOri.ResolutionPlani().y;
                Pterr.z = aMntImg->getr(Pmnt,NoData)*aMntOri.ResolutionAlti() + aMntOri.OrigineAlti();
                // Position dans l'image
                Pt2dr Pimg = aCamera->R3toF2(Pterr);
                
                if ((Pimg.x<0)||(Pimg.y<0)||(Pimg.x>=ImgSz.x)||(Pimg.y>=ImgSz.y))
                    continue;
                
                // On regarde si le pixel demande est dans le buffer en memoire
                if ((vBuffer.size()==0)||
                    ((Pimg.x<bufferMin.x)||(Pimg.x>=bufferMax.x)||(Pimg.y<bufferMin.y)||(Pimg.y>=bufferMax.y)))
                {
                    unsigned int minx = (unsigned int)std::max(0,(int)Pimg.x-margeBuffer);
                    unsigned int miny = (unsigned int)std::max(0,(int)Pimg.y-margeBuffer);
                    unsigned int maxx = (unsigned int)std::min((int)ImgSz.x ,(int)minx + tailleBuffer + 2*margeBuffer);
                    unsigned int maxy = (unsigned int)std::min((int)ImgSz.y ,(int)miny + tailleBuffer + 2*margeBuffer);
                    bufferMin.x = minx;
                    bufferMin.y = miny;
                    bufferMax.x = maxx;
                    bufferMax.y = maxy;
					//Pt2di SzBuffer = bufferMax-bufferMin;
                    // on purge
                    for(size_t i=0;i<vBuffer.size();++i)
                    {
                        delete vBuffer[i];
                    }
                    vBuffer.clear();
					if (((maxx-minx)>(unsigned int)0)&&((maxy-miny)>(unsigned int)0))
                    {
                        vBuffer = createVTIm2DFromFile<U_INT2,INT4>(aNameFileImage,bufferMin,bufferMax-bufferMin);
                    }
                }
                if ((vBuffer.size()!=0)&&
                    ((Pimg.x>bufferMin.x)&&(Pimg.x<bufferMax.x)&&(Pimg.y>bufferMin.y)&&(Pimg.y<bufferMax.y)))
                {
                    // on peut utiliser le buffer en memoire
                    for(size_t i=0;i<vBuffer.size();++i)
                    {
                        double radio = vBuffer[i]->getr(Pt2dr(Pimg.x-bufferMin.x,Pimg.y-bufferMin.y),NoData);
                        vPtrOrtho[i]->oset(Portho,(int)radio);
                    }
                }
            }
        }
        std::cout << "Fin du remplissage de l ortho"<<std::endl;
        // Sauvegarde

        if (vPtrOrtho.size()==1)
        {
            Tiff_Im out(aNameResult.c_str(), vPtrOrtho[0]->sz(),GenIm::u_int2,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);
            ELISE_COPY(vPtrOrtho[0]->_the_im.all_pts(),vPtrOrtho[0]->_the_im.in(),out.out());
        }
        else if (vPtrOrtho.size()==3)
        {
            Tiff_Im out(aNameResult.c_str(), vPtrOrtho[0]->sz(),GenIm::u_int2,Tiff_Im::No_Compr,Tiff_Im::RGB,Tiff_Im::Empty_ARG);
            ELISE_COPY(vPtrOrtho[0]->_the_im.all_pts(),Virgule(vPtrOrtho[0]->_the_im.in(),vPtrOrtho[1]->_the_im.in(),vPtrOrtho[2]->_the_im.in()),out.out());
        }
        
        // Liberation de la memoire
        for(size_t i=0;i<vPtrOrtho.size();++i)
        {
            delete vPtrOrtho[i];
        }
        vPtrOrtho.clear();
        for(size_t i=0;i<vBuffer.size();++i)
        {
            delete vBuffer[i];
        }
        vBuffer.clear();
    }
    else
    {
        int tailleDalle = 5000;
        std::cout << "Il faut daller le traitement"<<std::endl;
        int NbX = SzOrtho.x / tailleDalle;
        if (NbX*tailleDalle < SzOrtho.x)
            ++NbX;
        int NbY = SzOrtho.y / tailleDalle;
        if (NbY*tailleDalle < SzOrtho.y)
            ++NbY;
        
		//int nbDalles = NbX*NbY;
        
        for(int nx = 0;nx<NbX;++nx)
        {
            for(int ny = 0;ny<NbY;++ny)
            {
                std::cout << "Traitement de la dalle : "<<nx<<" x "<<ny<<std::endl;
                int cmin = nx*tailleDalle;
                int lmin = ny*tailleDalle;
                int cmax = std::min(cmin+tailleDalle,SzOrtho.x);
                int lmax = std::min(lmin+tailleDalle,SzOrtho.y);
                Pt2di SzDalleOrtho(cmax-cmin,lmax-lmin);
                Pt2dr P0DalleOrtho;
                P0DalleOrtho.x = P0Ortho.x + cmin*resolution;
                P0DalleOrtho.y = P0Ortho.y - lmin*resolution;
                
                std::ostringstream oss;
                oss << baseName(aNameResult)<<"_"<<nx<<"x"<<ny;
                std::string aNameDalle = oss.str();
                
                // Creation de l'image
                std::vector<TIm2D<U_INT2,INT4>*> vPtrOrtho;
				for(size_t c=0;c<(unsigned int)ImgNbC;++c)
                {
                    vPtrOrtho.push_back(new TIm2D<U_INT2,INT4>(SzDalleOrtho));
                }
                
                
                // Remplissage de l'image
                std::cout << "Debut du remplissage de l ortho"<<std::endl;
                for(int l=0;l<SzDalleOrtho.y;++l)
                {
                    for(int c=0;c<SzDalleOrtho.x;++c)
                    {
                        Pt2di Portho(c,l);
                        Pt3dr Pterr;
                        Pterr.x = P0DalleOrtho.x + c * resolution;
                        Pterr.y = P0DalleOrtho.y - l * resolution;
                        // Position dans le MNT
                        Pt2dr Pmnt;
                        Pmnt.x = (Pterr.x-aMntOri.OriginePlani().x)/aMntOri.ResolutionPlani().x;
                        Pmnt.y = (Pterr.y-aMntOri.OriginePlani().y)/aMntOri.ResolutionPlani().y;
                        Pterr.z = aMntImg->getr(Pmnt,NoData)*aMntOri.ResolutionAlti() + aMntOri.OrigineAlti();
                        // Position dans l'image
                        Pt2dr Pimg = aCamera->R3toF2(Pterr);
                        
                        if ((Pimg.x<0)||(Pimg.y<0)||(Pimg.x>=ImgSz.x)||(Pimg.y>=ImgSz.y))
                            continue;
                        
                        // On regarde si le pixel demande est dans le buffer en memoire
                        if ((vBuffer.size()==0)||
                            ((Pimg.x<bufferMin.x)||(Pimg.x>=bufferMax.x)||(Pimg.y<bufferMin.y)||(Pimg.y>=bufferMax.y)))
                        {
                            unsigned int minx = (unsigned int)std::max(0,(int)Pimg.x-margeBuffer);
                            unsigned int miny = (unsigned int)std::max(0,(int)Pimg.y-margeBuffer);
                            unsigned int maxx = (unsigned int)std::min((int)ImgSz.x ,(int)minx + tailleBuffer + 2*margeBuffer);
                            unsigned int maxy = (unsigned int)std::min((int)ImgSz.y ,(int)miny + tailleBuffer + 2*margeBuffer);
							if (((maxx-minx)>(unsigned int)0)&&((maxy-miny)>(unsigned int)0))
                            {
                                bufferMin.x = minx;
                                bufferMin.y = miny;
                                bufferMax.x = maxx;
                                bufferMax.y = maxy;
								//Pt2di SzBuffer = bufferMax-bufferMin;
                                // on purge
                                for(size_t i=0;i<vBuffer.size();++i)
                                {
                                    delete vBuffer[i];
                                }
                                vBuffer.clear();
                                vBuffer = createVTIm2DFromFile<U_INT2,INT4>(aNameFileImage,bufferMin,bufferMax-bufferMin);
                                std::cout << ".";
                            }
                        }
                        if ((vBuffer.size()!=0)&&
                            ((Pimg.x>bufferMin.x)&&(Pimg.x<bufferMax.x)&&(Pimg.y>bufferMin.y)&&(Pimg.y<bufferMax.y)))
                        {
                            // on peut utiliser le buffer en memoire
                            for(size_t i=0;i<vBuffer.size();++i)
                            {
                                double radio = vBuffer[i]->getr(Pt2dr(Pimg.x-bufferMin.x,Pimg.y-bufferMin.y),NoData);
                                vPtrOrtho[i]->oset(Portho,(int)radio);
                            }
                        }
                    }
                }
                std::cout << "Fin du remplissage de l ortho"<<std::endl;
                // Sauvegarde
                
                if (vPtrOrtho.size()==1)
                {
                    Tiff_Im out((aNameDalle+".tif").c_str(), vPtrOrtho[0]->sz(),GenIm::u_int2,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);
                    ELISE_COPY(vPtrOrtho[0]->_the_im.all_pts(),vPtrOrtho[0]->_the_im.in(),out.out());
                }
                else if (vPtrOrtho.size()==3)
                {
                    Tiff_Im out((aNameDalle+".tif").c_str(), vPtrOrtho[0]->sz(),GenIm::u_int2,Tiff_Im::No_Compr,Tiff_Im::RGB,Tiff_Im::Empty_ARG);
                    ELISE_COPY(vPtrOrtho[0]->_the_im.all_pts(),Virgule(vPtrOrtho[0]->_the_im.in(),vPtrOrtho[1]->_the_im.in(),vPtrOrtho[2]->_the_im.in()),out.out());
                }
                
                //TFW
                {
                    
                    cFileOriMnt aMntOri;
                    
                    aMntOri.OriginePlani()=P0DalleOrtho;
                    aMntOri.ResolutionPlani()=Pt2dr(resolution,-resolution);
                    aMntOri.NameFileMnt()=aNameDalle;
                    aMntOri.NombrePixels()=SzDalleOrtho;
                    
                    GenTFW(aMntOri,(aNameDalle+".tfw").c_str());

                }
                
                // Liberation de la memoire
                for(size_t i=0;i<vPtrOrtho.size();++i)
                {
                    delete vPtrOrtho[i];
                }
                vPtrOrtho.clear();
                for(size_t i=0;i<vBuffer.size();++i)
                {
                    delete vBuffer[i];
                }
                vBuffer.clear();
            }
        }
        
    }
    return EXIT_SUCCESS;
}


int ServiceGeoSud_TP2GCP_main(int argc, char **argv) {


    std::string aNameFileMNT;// un fichier xml de type FileOriMnt pour le MNT
    std::string aNameFileOrtho;// un fichier xml de type FileOriMnt pour l'ortho
    std::string aNameFilePOIOrtho;// un fichier dat de POI Ortho

    std::string aNameFileGrid;// un fichier GRID
    std::string aNameFileImage;// un fichier image
    std::string aNameFilePOIImage;// un fichier POI pour l'image associee a la GRID

    double seuilPixel=20;
    double seuilCorrel=0.7;
    int SzW=10;
    std::string aNameResult="result.txt";// un fichier resultat

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aNameFileMNT,"xml file (FileOriMnt) for the DTM")
                   << EAMC(aNameFileOrtho,"xml file (FileOriMnt) for the Ortho")
                   << EAMC(aNameFilePOIOrtho,"POI file for the Ortho")
                    << EAMC(aNameFileGrid,"GRID File")
                    << EAMC(aNameFileImage,"Image File")
                    << EAMC(aNameFilePOIImage,"POI file (related with the GRID and the Image)"),
     LArgMain() << EAM(seuilCorrel,"CorMin",true,"correl threshold")
                << EAM(seuilPixel,"Dmax",true,"distance threshlod (in pixel)")
                << EAM(SzW,"SzW",true,"half window size (in pixel)")
     << EAM(aNameResult,"Output","output file name")
     );

    return TP2GCP(aNameFileMNT,aNameFileOrtho,aNameFilePOIOrtho,aNameFileGrid,aNameFileImage,aNameFilePOIImage,seuilPixel,seuilCorrel,SzW,aNameResult);
}

int ServiceGeoSud_Ortho_main(int argc, char **argv) {


    std::string aNameFileMNT;// un fichier xml de type FileOriMnt pour le MNT

    std::string aNameFileGrid;// un fichier GRID
    std::string aNameFileImage;// un fichier image
    std::string aNameResult;// un fichier resultat

    double resolution;

    ElInitArgMain
    (
     argc, argv,
     LArgMain() << EAMC(aNameFileMNT,"xml file (FileOriMnt) for the DTM")
     << EAMC(aNameFileGrid,"GRID File")
     << EAMC(aNameFileImage,"Image File")
     << EAMC(resolution,"ground resolution")
     << EAMC(aNameResult,"Output file name"),
     LArgMain()
     );

    return  Ortho(aNameFileMNT,aNameFileGrid,aNameFileImage,resolution,aNameResult);
}

int ServiceGeoSud_Surf_main(int argc, char **argv){
    std::string aFullName;
    std::string aNameOut;

    int octaves=5;
    int intervals=4;
    int init_samples=2;
    int nbPoints=0;

    ElInitArgMain
    (
     argc, argv,
     LArgMain() << EAMC(aFullName,"Input filename")
     << EAMC(aNameOut,"output filename") ,
     LArgMain()
     << EAM(octaves,"Octaves",true,"Octaves")
     << EAM(intervals,"intervals",true,"Intervals")
     << EAM(init_samples,"init_samples",true,"init_samples")
     << EAM(nbPoints,"nbPoints",true,"nbPoints")
     );

    Pt2di ImgSz = getImageSize(aFullName);

    std::cout << "Taille de l'image  : "<<ImgSz.x<<" x "<<ImgSz.y<<std::endl;
    if (nbPoints == 0)
    {
        nbPoints = sqrt((float)ImgSz.x*ImgSz.y);
    }
    int tailleDalle = 4000;
    int NbX = ImgSz.x / tailleDalle;
    if (NbX*tailleDalle < ImgSz.x)
        ++NbX;
    int NbY = ImgSz.y / tailleDalle;
    if (NbY*tailleDalle < ImgSz.y)
        ++NbY;
    list<DigeoPoint> total_list;


    int nbDalles = NbX*NbY;
    int nbPointsParDalle = std::max(nbPoints / nbDalles,10);

    for(int nx = 0;nx<NbX;++nx)
    {
        for(int ny = 0;ny<NbY;++ny)
        {
            std::cout << "Traitement de la dalle : "<<nx<<" x "<<ny<<std::endl;
            int cmin = nx*tailleDalle;
            int lmin = ny*tailleDalle;
            int cmax = std::min(cmin+tailleDalle,ImgSz.x);
            int lmax = std::min(lmin+tailleDalle,ImgSz.y);
            std::cout << "Crop : "<<cmin<<" "<<lmin<<" "<<cmax<<" "<<lmax<<std::endl;

            Pt2di PminCrop(cmin,lmin);
            Pt2di SzCrop(cmax-cmin,lmax-lmin);
            std_unique_ptr<TIm2D<U_INT2,INT4> > cropImg(createTIm2DFromFile<U_INT2,INT4>(aFullName,PminCrop,SzCrop));
            if (cropImg.get()==NULLPTR)
            {
                cerr << "Error in "<<aFullName<<" Crop : "<<PminCrop.x<<" "<<PminCrop.y<<" / "<<SzCrop.x<<" "<<SzCrop.y<<std::endl;
                return EXIT_FAILURE;
            }


            BufferImage<unsigned short> aBuffer(cmax-cmin,lmax-lmin,1,cropImg->_the_im.data_lin(),1,(cmax-cmin),1);

            Surf s(aBuffer,octaves,intervals,init_samples,nbPointsParDalle);
            std::cout << "Nombre de points : "<<s.vPoints.size()<<std::endl;
            for(size_t i=0;i<s.vPoints.size();++i)
            {
                SurfPoint const &surfPt =s.vPoints[i];
                //std::cout << "Point : "<<surfPt.x()+cmin<<" "<<surfPt.y()+lmin<<" "<<surfPt.descripteur.size()<<std::endl;
                DigeoPoint pt;
                pt.x =surfPt.x()+cmin;
                pt.y =surfPt.y()+lmin;

                REAL8* des = new REAL8[DIGEO_DESCRIPTOR_SIZE];
                for(int d=0;d<DIGEO_DESCRIPTOR_SIZE;++d)
                {
                    if (d< (int) surfPt.descripteur.size())
                    {
                        des[d] = surfPt.descripteur[d];
                    }
                    else
                    {
                        des[d]=0.;
                    }
                }
                pt.addDescriptor(0.,des);
                delete[] des;
                total_list.push_back(pt);
            }

        }
    }

    cout << total_list.size() << " points" << endl;
    DigeoPoint::writeDigeoFile(aNameOut, total_list);


    // Verification
    {
        // Chargement des points d'interet dans l'ortho
        vector<DigeoPoint> vPts;

        DigeoPoint::readDigeoFile( aNameOut, true, vPts );
        std::cout << "Nombre de points lus: "<<vPts.size()<<std::endl;

    }

    return 0;
}


int debug(int argc, char **argv)
{
    std::string nomDalleOrtho("Dalle_0x0_ortho.tif");
    // Chargement de l'ortho
    std_unique_ptr<TIm2D<U_INT1,INT4> > OrthoImg(createTIm2DFromFile<U_INT1,INT4>(nomDalleOrtho));
    std::cout << "Chargement de OrthoImg : "<<nomDalleOrtho<<std::endl;
    if (OrthoImg.get()==NULLPTR)
    {
        cerr << "Error in "<<nomDalleOrtho<<std::endl;
        return EXIT_FAILURE;
    }
    int SzW=10;
    double NoData = -9999.;

    DigeoPoint ptSift;
    ptSift.x = 3090.585959;
    ptSift.y = 245.973652;

    // Crop
    TIm2D<U_INT2,INT4> debugFenOrtho(Pt2di(2*SzW+1,2*SzW+1));
    std::vector<double> fenOrtho;
    for(int l=-SzW;l<=SzW;++l)
    {
        for(int c=-SzW;c<=SzW;++c)
        {
            double radio =OrthoImg->getr(Pt2dr((double)c+ptSift.x ,(double)l+ptSift.y),NoData);
            fenOrtho.push_back(radio);
            std::cout << "Point "<<(double)c+ptSift.x<<" "<<(double)l+ptSift.y<<" -> "<<c+SzW<<" "<<l+SzW<<" : "<<radio<<std::endl;
            debugFenOrtho.oset(Pt2di(c+SzW,l+SzW),(int)radio);
        }
    }
    // Sauvegarde
    Tiff_Im debugFenOrtho_out("debug_ortho_bis.tif", debugFenOrtho.sz(),GenIm::u_int2,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);
    ELISE_COPY(debugFenOrtho._the_im.all_pts(),debugFenOrtho._the_im.in(),debugFenOrtho_out.out());
    return 0;
}

int ServiceGeoSud_GeoSud_main(int argc, char **argv){
    //return debug(argc,argv);

    std::string aFullName;
    std::string aKeyGPP;
    std::string aHttpProxy;
    std::string aGRIDExt("GRI");
    std::string aFileMnt;
    bool aExportMM = false;

    double ZMoy = 0.;
    double seuilPixel=20;
    double seuilCorrel=0.7;
    int SzW=10;

    ElInitArgMain
    (
         argc, argv,
         LArgMain() << EAMC(aFullName,"Full Name (Dir+Pat)")
                    << EAMC(aKeyGPP,"GPP Key"),
         LArgMain()
                    << EAM(aGRIDExt,"Grid",true,"GRID file extension")
                    << EAM(aFileMnt,"Mnt",true,"xml file (FileOriMnt) for the DTM")
                    << EAM(aHttpProxy,"Proxy",true,"http proxy for GPP access")
                    << EAM(aExportMM,"ExportMM", true, "export GCP and tie-points in MicMac xml format (def=false)")
    );

    //double seuilPixel2 = pow(seuilPixel,2);

    std::ofstream ficPtLiaison("POINTS_LIAISON.TXT");
    ficPtLiaison << "# num_pt alti correc_alti prec_alti Actif/Inact"<<std::endl;
    int idPtLiaison = 1;

    std::ofstream ficLiaisons("LIAISONS.TXT");
    ficLiaisons << "# num_pt(rang fic_liaison)) num_modele ligne colonne prec_lig(m) prec_col(m) Actif/Inact"<<std::endl;

    std::ofstream ficModeles("MODELES.TXT");

    std::ofstream ficAmers("AMERS.TXT");
    ficAmers << "# num_pt X Y alti correc_lon correc_lat correc_alti prec_lon(m) prec_lat(m) prec_alt(m) Actif/Inact"<<std::endl;
    int idAmer = 1;

    std::ofstream ficAppuis("APPUIS.TXT");
    ficAppuis << "# num_pt(rang fic_amer) num_modele ligne colonne prec_lig(m) prec_col(m) Actif/Inact"<<std::endl;

    std::string aDir,aPat;
    SplitDirAndFile(aDir,aPat,aFullName);
    std::list<std::string> aLFile;
    cInterfChantierNameManipulateur *aICNM;
    aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    aLFile = aICNM->StdGetListOfFile(aPat);

    std::cout << "Nombre de fichiers a traiter : "<<aLFile.size()<<std::endl;

    // On cherche l'emprise du chantier
    double xminChantier,yminChantier,xmaxChantier,ymaxChantier;
    xminChantier = yminChantier = xmaxChantier = ymaxChantier = 0;
    bool first = true;

    std::list<std::string>::const_iterator it,fin=aLFile.end();
    std::list<std::string> aLFilePoi;
    std::list<std::string> aLFileGrid;
    std::list<ElCamera*> aLCamera;
    std::list<std::vector<DigeoPoint> > aLPoi;
    for(it=aLFile.begin();it!=fin;++it)
    {
        std::string aNameFileImage = (*it);
        std::cout << "fichier image : "<<aNameFileImage<<std::endl;

        // taille de l'image
        Pt2di ImgSz = getImageSize(aNameFileImage);

        // On cherche la grille correspondante
        int placePoint = -1;
        for(int l=(int)(aNameFileImage.size()-1);(l>=0)&&(placePoint==-1);--l)
        {
            if (aNameFileImage[l]=='.')
            {
                placePoint = l;
            }
        }
        //std::string ext = std::string("");
        if (placePoint!=-1)
        {
            std::string baseName;
            baseName.assign(aNameFileImage.begin(),aNameFileImage.begin()+placePoint+1);
            std::string modelName;
            modelName.assign(aNameFileImage.begin(),aNameFileImage.begin()+placePoint);
            std::string aNameFileGrid = baseName+aGRIDExt;
            std::string aNameFilePOI = baseName+"dat";

            ficModeles << modelName <<" "<<aLCamera.size()+1<<" 1"<<std::endl;

            aLFilePoi.push_back(aNameFilePOI);
            aLFileGrid.push_back(aNameFileGrid);

            std::cout << "fichier GRID : "<<aNameFileGrid<<std::endl;
            std::cout << "fichier POI : "<<aNameFilePOI<<std::endl;

            // Chargement de la grille et de l'image
            ElAffin2D oriIntImaM2C;
            std_unique_ptr<ElCamera> aCamera(new cCameraModuleOrientation(new OrientationGrille(aNameFileGrid),ImgSz,oriIntImaM2C));


            // On cherche l'emprise de l'image
            double xmin,ymin,xmax,ymax;
            // Projection des coins de l'image pour trouver l'emprise
            {
                Pt3dr Pterr = aCamera->F2AndZtoR3(Pt2dr(0,0),ZMoy);
                xmin = Pterr.x;
                ymin = Pterr.y;
                xmax = Pterr.x;
                ymax = Pterr.y;
            }
            {
                Pt3dr Pterr = aCamera->F2AndZtoR3(Pt2dr(ImgSz.x,0),ZMoy);
                if (xmin>Pterr.x)
                    xmin = Pterr.x;
                else if (xmax<Pterr.x)
                    xmax = Pterr.x;
                if (ymin>Pterr.y)
                    ymin = Pterr.y;
                else if (ymax<Pterr.y)
                    ymax = Pterr.y;
            }
            {
                Pt3dr Pterr = aCamera->F2AndZtoR3(Pt2dr(ImgSz.x,ImgSz.y),ZMoy);
                if (xmin>Pterr.x)
                    xmin = Pterr.x;
                else if (xmax<Pterr.x)
                    xmax = Pterr.x;
                if (ymin>Pterr.y)
                    ymin = Pterr.y;
                else if (ymax<Pterr.y)
                    ymax = Pterr.y;
            }
            {
                Pt3dr Pterr = aCamera->F2AndZtoR3(Pt2dr(0,ImgSz.y),ZMoy);
                if (xmin>Pterr.x)
                    xmin = Pterr.x;
                else if (xmax<Pterr.x)
                    xmax = Pterr.x;
                if (ymin>Pterr.y)
                    ymin = Pterr.y;
                else if (ymax<Pterr.y)
                    ymax = Pterr.y;
            }
            std::cout << "Emprise Terrain de l'image : "<<xmin<<" "<<ymin<<" "<<xmax<<" "<<ymax<<std::endl;
            if (first)
            {
                first = false;
                xminChantier = xmin;
                xmaxChantier = xmax;
                yminChantier = ymin;
                ymaxChantier = ymax;
            }
            else
            {
                if (xmin<xminChantier)
                    xminChantier=xmin;
                if (xmax>xmaxChantier)
                    xmaxChantier=xmax;
                if (ymin<yminChantier)
                    yminChantier=ymin;
                if (ymax>ymaxChantier)
                    ymaxChantier=ymax;
            }

            FILE* fPoi = fopen(aNameFilePOI.c_str(),"r");
            if (fPoi == NULL)
            {
                std::cout << "Le fichier POI n'existe pas encore"<<std::endl;
                // Extraction des POI
                {
                    std::string cmdPOI="mm3d TestLib Surf "+aNameFileImage+" "+aNameFilePOI;
                    system(cmdPOI.c_str());
                }
            }
            else
            {
                std::cout << "Le fichier Poi "<<aNameFilePOI<<" existe deja"<<std::endl;
                fclose(fPoi);
            }



            // Chargement des points d'interet dans l'image
            vector<DigeoPoint> keypointsImage;

            if ( !DigeoPoint::readDigeoFile( aNameFilePOI, true, keypointsImage ) ){
                cerr << "WARNING: unable to read keypoints in [" << aNameFilePOI << "]" << endl;
                return EXIT_FAILURE;
            }

            aLPoi.push_back(keypointsImage);
            aLCamera.push_back(aCamera.release());
        }
    }

    // On arrondit
    xminChantier = (int)(xminChantier-1);
    xmaxChantier = (int)(xmaxChantier+1);
    yminChantier = (int)(yminChantier-1);
    ymaxChantier = (int)(ymaxChantier+1);



    std::string gppAccess;
    {
        std::ostringstream oss;
        oss << g_externalToolHandler.get( "curl" ).callName() << " -H='Referer: http://localhost' ";
        if (!aHttpProxy.empty())
            oss << "-x "<<aHttpProxy;
        gppAccess = oss.str();
    }

    // Chargement du MNT
    cFileOriMnt aMntOri;
    if (!aFileMnt.empty())
    {
        aMntOri=  StdGetFromPCP(aFileMnt,FileOriMnt);
    }
    else
    {
        // Chargement depuis le Geoportail
        double resolutionMnt=25;
        int NCmnt = (xmaxChantier-xminChantier)/resolutionMnt + 1;
        int NLmnt = (ymaxChantier-yminChantier)/resolutionMnt + 1;
        std::ostringstream oss;

        //wget http://wxs.ign.fr -e use_proxy=yes -e http_proxy=http://relay-gpp3-i-interco.sca.gpp.priv.atos.fr:3128{quote}
        //curl -o ./mnt_tmp2.tif -H="Referer: http://localhost" -x http://relay-gpp3-i-interco.sca.gpp.priv.atos.fr:3128 "http://wxs-i.ign.fr/7gr31kqe5xttprd2g7zbkqgo/geoportail/r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=ELEVATION.ELEVATIONGRIDCOVERAGE&STYLES=normal&FORMAT=image/geotiff&BBOX=43.278259,3.103180,43.411264,3.366021&CRS=EPSG:4326&WIDTH=500&HEIGHT=350"

        oss << std::fixed << gppAccess<<" -o mnt_25m.tif  \"http://wxs-i.ign.fr/"<<aKeyGPP<<"/geoportail/r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=ELEVATION.ELEVATIONGRIDCOVERAGE&STYLES=normal&FORMAT=image/geotiff&BBOX="<< xminChantier<<","<<yminChantier<<","<<xminChantier+NCmnt*resolutionMnt<<","<<yminChantier+NLmnt*resolutionMnt<<"&CRS=EPSG:2154&WIDTH="<<NCmnt<<"&HEIGHT="<<NLmnt<<"\"";

        //oss << std::fixed << "curl -o mnt_25m.tif -H='Referer: http://localhost' \"http://wxs-i.ign.fr/"<<aKeyGPP<<"/geoportail/r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=ELEVATION.ELEVATIONGRIDCOVERAGE&STYLES=normal&FORMAT=image/geotiff&BBOX="<< xminChantier<<","<<yminChantier<<","<<xminChantier+NCmnt*resolutionMnt<<","<<yminChantier+NLmnt*resolutionMnt<<"&CRS=EPSG:2154&WIDTH="<<NCmnt<<"&HEIGHT="<<NLmnt<<"\"";
        std::cout << "commande : "<<oss.str()<<std::endl;
        system(oss.str().c_str());
        aMntOri.OriginePlani()=Pt2dr(xminChantier,ymaxChantier);
        aMntOri.ResolutionPlani()=Pt2dr(resolutionMnt,-resolutionMnt);
        aMntOri.NameFileMnt()=std::string("mnt_25m.tif");
        aMntOri.NombrePixels()=Pt2di(NCmnt,NLmnt);
        aMntOri.OrigineAlti()=0;
        aMntOri.ResolutionAlti()=1.;
    }
    std::cout << "Taille du MNT : "<<aMntOri.NombrePixels().x<<" "<<aMntOri.NombrePixels().y<<std::endl;
    std_unique_ptr<TIm2D<REAL4,REAL8> > aMntImg(createTIm2DFromFile<REAL4,REAL8>(aMntOri.NameFileMnt()));
    if (aMntImg.get()==NULLPTR)
    {
        cerr << "Error in "<<aMntOri.NameFileMnt()<<std::endl;
        return EXIT_FAILURE;
    }

    // Calcul des points entre les images avec Ann
    std::list<ElCamera*>::iterator itCamera = aLCamera.begin();
    int numImg1 = 1;
    for(it=aLFilePoi.begin();it!=aLFilePoi.end();++it)
    {
        std::list<std::string>::const_iterator it2;
        ElCamera* aCamera = (*itCamera);
        int numImg2=1;
        for(it2=aLFilePoi.begin();it2!=it;++it2)
        {
            std::string nomRes = "out.res";
            std::string cmdAnn="mm3d Ann "+(*it)+" "+(*it2)+" out.res";
            system(cmdAnn.c_str());
            // Il faut exporter ces points de liaison en allant chercher le Z sur un MNT basse resolution (a 25m)
            std::ifstream fic(nomRes.c_str());
            while(fic.good())
            {
                double c1,l1,c2,l2;
                fic >> c1 >> l1 >> c2 >> l2;
                if (fic.good())
                {
                    //std::cout << "Point de liaison "<<c1<<" "<<l1<<" | "<<c2<<" "<<l2<<std::endl;
                    // On estime le Z
                    Pt3dr Pt3D = Img2Terrain(aCamera,aMntImg.get(),aMntOri,ZMoy,Pt2di(c1,l1));
                    // On exporte le point
                    std::cout << "Point de liaison "<<c1<<" "<<l1<<" | "<<c2<<" "<<l2<<" | "<<Pt3D.z<<std::endl;

                    ficPtLiaison << idPtLiaison<<" "<<Pt3D.z << " 0.0 200 1"<<std::endl;
                    ficLiaisons << idPtLiaison << " "<<numImg1<<" "<<l1<<" "<<c1<<" 5.00e-01  5.00e-01 1"<<std::endl;
                    ficLiaisons << idPtLiaison << " "<<numImg2<<" "<<l2<<" "<<c2<<" 5.00e-01  5.00e-01 1"<<std::endl;
                    ++idPtLiaison;
                }
            }
            ++numImg2;
        }
        ++itCamera;
        ++numImg1;
    }



    // On arrondit
    xminChantier = (int)(xminChantier-1);
    xmaxChantier = (int)(xmaxChantier+1);
    yminChantier = (int)(yminChantier-1);
    ymaxChantier = (int)(ymaxChantier+1);
    /*
    xminChantier = 375000;
    xmaxChantier = 379000;
    yminChantier = 6570000;
    ymaxChantier = 6574000;
    */

    double resolution = 1.;//1m
    int NC = (xmaxChantier-xminChantier)/resolution;
    int NL = (ymaxChantier-yminChantier)/resolution;

    std::cout << std::fixed << "Emprise du chantier : "<<xminChantier<<" "<<yminChantier<<" "<<xmaxChantier<<" "<<ymaxChantier<<std::endl;

    // Si le chantier est trop grand, il faut daller
    int tailleDalle = 4000;
    int NbX = NC/tailleDalle;
    if (NbX*tailleDalle<NC)
        ++NbX;
    int NbY = NL/tailleDalle;
    if (NbY*tailleDalle<NL)
        ++NbY;
    std::cout << "Nombre de dalles : "<<NbX<<" x "<<NbY<<std::endl;
    for(int c=0;c<NbX;++c)
    // Extraction de l'ortho
    {
        for(int l=0;l<NbY;++l)
        {
            std::cout << "Traitement de la dalle "<<c<<" "<<l<<std::endl;
            std::ostringstream oss ;
            oss << "Dalle_"<<c<<"x"<<l;
            std::string nomDalle = oss.str();
            std::string nomDalleOrtho = nomDalle+"_ortho.tif";
            //std::string nomDalleMntBil = nomDalle+"_mnt.bil";
            //std::string nomDalleMntHdr = nomDalle+"_mnt.hdr";
            std::string nomDallePoi = nomDalle+"_ortho.dat";

            int ncDalle = std::min(tailleDalle,NC-c*tailleDalle);
            int nlDalle = std::min(tailleDalle,NL-l*tailleDalle);
            double xminDalle = xminChantier + c*tailleDalle*resolution;
            double yminDalle = yminChantier + l*tailleDalle*resolution;
            double xmaxDalle = xminDalle + ncDalle*resolution;
            double ymaxDalle = yminDalle + nlDalle*resolution;
            // Extraction de l'ortho si necessaire
            FILE* fDalleOrtho = fopen(nomDalleOrtho.c_str(),"r");
            if (fDalleOrtho == NULL)
            {
                std::ostringstream oss;
                oss << std::fixed << gppAccess << " -o "<<nomDalleOrtho<< " \"http://wxs-i.ign.fr/"<<aKeyGPP<<"/geoportail/r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=ORTHOIMAGERY.ORTHOPHOTOS&STYLES=normal&FORMAT=image/geotiff&BBOX="<< xminDalle<<","<<yminDalle<<","<<xmaxDalle<<","<<ymaxDalle<<"&CRS=EPSG:2154&WIDTH="<<ncDalle<<"&HEIGHT="<<nlDalle<<"\"";
                //oss << std::fixed << "curl -o "<<nomDalleOrtho<<" -H='Referer: http://localhost' \"http://wxs-i.ign.fr/"<<aKeyGPP<<"/geoportail/r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=ORTHOIMAGERY.ORTHOPHOTOS&STYLES=normal&FORMAT=image/geotiff&BBOX="<< xminDalle<<","<<yminDalle<<","<<xmaxDalle<<","<<ymaxDalle<<"&CRS=EPSG:2154&WIDTH="<<ncDalle<<"&HEIGHT="<<nlDalle<<"\"";

                /*
                oss << std::fixed << g_externalToolHandler.get( "curl" ).callName() + " -o "<<nomDalleOrtho<<" -H='Referer: http://localhost' \"http://wxs-i.ign.fr/"<<aKeyGPP<<"/geoportail/r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=ORTHOIMAGERY.ORTHOPHOTOS&STYLES=normal&FORMAT=image/geotiff&BBOX="<< xminDalle<<","<<yminDalle<<","<<xmaxDalle<<","<<ymaxDalle<<"&CRS=EPSG:2154&WIDTH="<<ncDalle<<"&HEIGHT="<<nlDalle<<"\"";
                */
                std::cout << "commande : "<<oss.str()<<std::endl;
                system(oss.str().c_str());
            }
            else
            {
                fclose(fDalleOrtho);
            }

            // Chargement de l'ortho
            std::cout << "Chargement de la dalle d'ortho: ..."<<std::endl;
            std_unique_ptr<TIm2D<U_INT1,INT4> > OrthoImg(createTIm2DFromFile<U_INT1,INT4>(nomDalleOrtho));
            std::cout << "Chargement de OrthoImg : "<<nomDalleOrtho<<std::endl;
            if (OrthoImg.get()==NULLPTR)
            {
                cerr << "Error in "<<nomDalleOrtho<<std::endl;
                return EXIT_FAILURE;
            }

            // On teste si l'image contient quelque chose (cas des bords de mer)
            bool empty = true;
            {
                double min,max;
                ELISE_COPY
                (
                 OrthoImg->_the_im.all_pts(),
                 OrthoImg->_the_im.in(),
                 VMin(min)
                 );
                ELISE_COPY
                (
                 OrthoImg->_the_im.all_pts(),
                 OrthoImg->_the_im.in(),
                 VMax(max)
                 );

                std::cout << "Min : "<<min<<" Max : "<<max<<std::endl;
                if (min!=max)
                    empty=false;
            }
            if (empty)
                continue;

            FILE* fDallePoi = fopen(nomDallePoi.c_str(),"r");
            // Extraction des POI
            if (fDallePoi==NULL)
            {
                std::string cmdPOI="mm3d TestLib Surf "+nomDalleOrtho+" "+nomDallePoi+ " nbPoints=100";
                system(cmdPOI.c_str());
            }
            else
            {
                fclose(fDallePoi);
            }
            // Extraction du MNT
            /*
            {
                std::ostringstream oss;
                oss << std::fixed << "curl -o "<<nomDalleMntBil<<" -H='Referer: http://localhost' \"http://wxs-i.ign.fr/"<<aKeyGPP<<"/geoportail/r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=ELEVATION.ELEVATIONGRIDCOVERAGE&STYLES=normal&FORMAT=image/x-bil;bits=32&BBOX="<< xminDalle<<","<<yminDalle<<","<<xmaxDalle<<","<<ymaxDalle<<"&CRS=EPSG:2154&WIDTH="<<ncDalle<<"&HEIGHT="<<nlDalle<<"\"";
                std::cout << "commande : "<<oss.str()<<std::endl;
                system(oss.str().c_str());


                //echo 'NROWS 675\nNCOLS 1769\nNBANDS 1\nBYTEORDER I\nNBITS 32\nLAYOUT  BIL\nSIGNE 1\nBAND_NAMES Z\n' > mnt.HDR
                std::ostringstream ossHdr;
                ossHdr << "echo 'NROWS "<<nlDalle<<"\nNCOLS "<<ncDalle<<"\nNBANDS 1\"nBYTEORDER I\nNBITS 32\nLAYOUT  BIL\nSIGNE 1\nBAND_NAMES Z\n' > "<<nomDalleMntHdr;
                system(ossHdr.str().c_str());
            }
             */

            // Chargement des points d'interet dans l'ortho
            vector<DigeoPoint> keypointsOrtho;

            if ( !DigeoPoint::readDigeoFile( nomDalle+"_ortho.dat", true, keypointsOrtho ) ){
                cerr << "WARNING: unable to read keypoints in [" << nomDalle+"_ortho.dat" << "]" << endl;
                return EXIT_FAILURE;
            }
            std::cout << "Nombre de points dans l'ortho : "<<keypointsOrtho.size()<<std::endl;

            double NoData = -9999.;

            // Export sous forme de dico appuis et mesures
            cSetOfMesureAppuisFlottants aSetDicoMesure;
            cDicoAppuisFlottant  aDicoAppuis;

            // on parcourt les points sift de l'ortho
            vector<DigeoPoint>::const_iterator itKP,finKP=keypointsOrtho.end();
            for(itKP=keypointsOrtho.begin();itKP!=finKP;++itKP)
            {
                DigeoPoint const &ptSift = (*itKP);
                // On estime la position 3D du point
                // ToDo: ajouter l'interpolation de l'alti dans le mnt
                Pt2dr ptOrtho(ptSift.x,ptSift.y);
                //std::cout << "Point img dans l'ortho : "<<ptOrtho.x<<" "<<ptOrtho.y<<std::endl;
                ptOrtho.x =  ptOrtho.x*resolution + xminDalle;
                ptOrtho.y =  -ptOrtho.y*resolution + ymaxDalle;
                //std::cout << "Point terrain 2D : "<<ptOrtho.x<<" "<<ptOrtho.y<<std::endl;
                // Position dans le MNT
                Pt2dr ptMnt;
                ptMnt.x = (ptOrtho.x-aMntOri.OriginePlani().x)/aMntOri.ResolutionPlani().x;
                ptMnt.y = (ptOrtho.y-aMntOri.OriginePlani().y)/aMntOri.ResolutionPlani().y;
                //std::cout << "Point img dans le Mnt : "<<ptMnt.x<<" "<<ptMnt.y<<std::endl;
                double alti = aMntImg->getr(ptMnt,NoData)*aMntOri.ResolutionAlti() + aMntOri.OrigineAlti();
                if (alti == NoData)
                {
                    std::cout << "Pas d'altitude trouvee pour le point : "<<ptOrtho.x<<" "<<ptOrtho.y<<std::endl;
                    std::cout << "On passe au point suivant"<<std::endl;
                    break;
                }
                //std::cout << "Altitude Mnt : "<<alti<<std::endl;
                // Position dans l'image
                Pt3dr pt3(ptOrtho.x,ptOrtho.y,alti);
                //std::cout << "Point terrain : "<<pt3.x<<" "<<pt3.y<<" "<<pt3.z<<std::endl;

                bool usePt = false;

                //TIm2D<U_INT2,INT4> debugFenOrtho(Pt2di(2*SzW+1,2*SzW+1));

                std::vector<double> fenOrtho;
                for(int l=-SzW;l<=SzW;++l)
                {
                    for(int c=-SzW;c<=SzW;++c)
                    {
                        double radio =OrthoImg->getr(Pt2dr((double)c+ptSift.x ,(double)l+ptSift.y),NoData);
                        fenOrtho.push_back(radio);
                        //std::cout << "Point "<<(double)c+ptSift.x<<" "<<(double)l+ptSift.y<<" -> "<<c+SzW<<" "<<l+SzW<<" : "<<radio<<std::endl;
                        //debugFenOrtho.oset(Pt2di(c+SzW,l+SzW),(int)radio);
                    }
                }
                // Sauvegarde
                //Tiff_Im debugFenOrtho_out("debug_ortho.tif", debugFenOrtho.sz(),GenIm::u_int2,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);
                //ELISE_COPY(debugFenOrtho._the_im.all_pts(),debugFenOrtho._the_im.in(),debugFenOrtho_out.out());



                int numImg = 1;
                std::list<std::string>::const_iterator itLF=aLFile.begin();
                std::list<std::string>::const_iterator itLFPoi=aLFilePoi.begin();
                std::list<std::vector<DigeoPoint> >::const_iterator itPoi=aLPoi.begin();
                itCamera = aLCamera.begin();
                for(it=aLFilePoi.begin();it!=aLFilePoi.end();++it,++itLF,++itLFPoi)
                {
                    ElCamera* aCamera = (*itCamera);
                    Pt2dr pImg = aCamera->R3toF2(pt3);
                    // taille de l'image
                    Pt2di ImgSz = getImageSize(*itLF);

                    //std::cout << "Point Image : "<<pImg.x<<" "<<pImg.y<<std::endl;

                    cMesureAppuiFlottant1Im aDicoMesure;
                    //cout << "Img name: " << *itLF << endl;
                    aDicoMesure.NameIm()= *itLF;

                    // Autre approche: on teste tout le voisinage
#if 1

                    // On prepare le crop
                    bool first=true;
                    double cmin = 0 ,cmax = 0 ,lmin = 0,lmax = 0;

                    for(int l=ptSift.y-SzW-seuilPixel;l<=(ptSift.y+SzW+seuilPixel);++l)
                    {
                        for(int c=ptSift.x-SzW-seuilPixel;c<=(ptSift.x+SzW+seuilPixel);++c)
                        {
                            Pt3dr P3D;
                            P3D.x = xminDalle + c * resolution;
                            P3D.y = ymaxDalle - l * resolution;
                            P3D.z = alti;

                            Pt2dr p2 = aCamera->R3toF2(P3D);
                            if (first)
                            {
                                first = false;
                                cmin = p2.x;
                                cmax = p2.x;
                                lmin = p2.y;
                                lmax = p2.y;
                            }
                            else
                            {
                                if (cmin>p2.x)
                                    cmin=p2.x;
                                else if (cmax<p2.x)
                                    cmax=p2.x;
                                if (lmin>p2.y)
                                    lmin=p2.y;
                                else if (lmax<p2.y)
                                    lmax=p2.y;
                            }
                        }
                    }

                    cmin-=1.;
                    lmin-=1.;
                    cmax+=1.;
                    lmax+=1.;

                    if (cmin<0)
                        cmin=0;
                    if (lmin<0)
                        lmin=0;
                    if (cmax>=ImgSz.x)
                        cmax=ImgSz.x-1;
                    if (lmax>=ImgSz.y)
                        lmax=ImgSz.y-1;

                    Pt2di PminCrop((int)round_ni(cmin),(int)round_ni(lmin));
                    Pt2di SzCrop((int)round_ni(cmax-PminCrop.x),(int)round_ni(lmax-PminCrop.y));
                    if( (SzCrop.x<=0)||(SzCrop.y<=0))
                        continue;

                    //std::cout << "Crop : "<<PminCrop.x<<" "<<PminCrop.y<<" / "<<SzCrop.x<<" "<<SzCrop.y<<std::endl;
                    std_unique_ptr<TIm2D<U_INT2,INT4> > cropImg(createTIm2DFromFile<U_INT2,INT4>((*itLF),PminCrop,SzCrop));
                    if (cropImg.get()==NULLPTR)
                    {
                        cerr << "Error in "<<(*itLF)<<" Crop : "<<PminCrop.x<<" "<<PminCrop.y<<" / "<<SzCrop.x<<" "<<SzCrop.y<<std::endl;
                        return EXIT_FAILURE;
                    }
                    // Sauvegarde
                    //Tiff_Im debugCropImg_out("debug_cropImg.tif", cropImg->sz(),GenIm::u_int2,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);
                    //ELISE_COPY(cropImg->_the_im.all_pts(),cropImg->_the_im.in(),debugCropImg_out.out());


                    double coefmax = -1.;
                    Pt2dr ptMaxCorrel;
                    for(int dl = -seuilPixel;dl<=seuilPixel;++dl)
                    {
                        for(int dc = -seuilPixel;dc<=seuilPixel;++dc)
                        {
                            Pt2dr pt2;
                            pt2.x = pImg.x + dc;
                            pt2.y = pImg.y + dl;
                            Pt3dr pTerrain = aCamera->F2AndZtoR3(Pt2dr(pt2.x,pt2.y),pt3.z);
                            std::vector<double> fenImage;
                            // Plutot que de reprojetter tous les points de la fenetre, on interpole la projection
                            Pt3dr pTerrain2;
                            pTerrain2.x = pTerrain.x + resolution;
                            pTerrain2.y = pTerrain.y;
                            pTerrain2.z = pTerrain.z;
                            Pt2dr ptImg2 = aCamera->R3toF2(pTerrain2);
                            double deltax_col = ptImg2.x - pt2.x;
                            double deltax_lig = ptImg2.y - pt2.y;

                            pTerrain2.x = pTerrain.x;
                            pTerrain2.y = pTerrain.y - resolution;
                            pTerrain2.z = pTerrain.z;
                            ptImg2 = aCamera->R3toF2(pTerrain2);
                            double deltay_col = ptImg2.x - pt2.x;
                            double deltay_lig = ptImg2.y - pt2.y;


                            //TIm2D<U_INT2,INT4> debugFenImg(Pt2di(2*SzW+1,2*SzW+1));

                            for(int dy=-SzW;dy<=SzW;++dy)
                            {
                                for(int dx=-SzW;dx<=SzW;++dx)
                                {
                                    Pt2dr p2;
                                    p2.x = pt2.x + dx * deltax_col + dy * deltay_col-PminCrop.x;
                                    p2.y = pt2.y + dx * deltax_lig + dy * deltay_lig-PminCrop.y;
                                    double radio =cropImg->getr(p2,NoData);
                                    fenImage.push_back(radio);
                                    //debugFenImg.oset(Pt2di(dx+SzW,dy+SzW),(int)radio);
                                }
                            }
                            // Coeff de correlation
                            double coef = correl(fenOrtho,fenImage,NoData);
                            //std::cout << "Correl : "<<coef<<std::endl;
                            if (coef>coefmax)
                            {
                                coefmax = coef;
                                ptMaxCorrel = pt2;
                                // Sauvegarde
                                //Tiff_Im debugFenImg_out("debug_img.tif", debugFenImg.sz(),GenIm::u_int2,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);
                                //ELISE_COPY(debugFenImg._the_im.all_pts(),debugFenImg._the_im.in(),debugFenImg_out.out());
                            }
                        }
                    }
                    //std::cout << "Max de correlation : "<<ptMaxCorrel.x<<" "<<ptMaxCorrel.y<<" correl = "<<coefmax<<std::endl;

                    if (coefmax >= seuilCorrel)
                    {
                        std::cout << "Point : "<<ptMaxCorrel.x<<" "<<ptMaxCorrel.y<<" Correl="<<coefmax<<std::endl;
                        ficAppuis << idAmer <<" "<<numImg<<" "<<ptMaxCorrel.y<<" "<<ptMaxCorrel.x<<" 5.00e-01  5.00e-01 0"<<std::endl;
                        usePt=true;

                        if (aExportMM)
                        {
                            cOneMesureAF1I aMesure;

                            aMesure.PtIm()=ptMaxCorrel;

                            std::ostringstream oss;
                            oss << idAmer;
                            aMesure.NamePt()=oss.str();

                            aDicoMesure.OneMesureAF1I().push_back(aMesure);
                        }
                    }
#else
                      // Chargement des points d'interet dans l'image
                    vector<DigeoPoint> const &keypointsImage=(*itPoi);


                    DigeoPoint ptHomoDmin;
                    Pt3dr pTerrainDmin;
                    double dmin = -1.;
                    // On cherche un POIImage proche de pImg
                    vector<DigeoPoint>::const_iterator it2,fin2=keypointsImage.end();
                    for(it2=keypointsImage.begin();it2!=fin2;++it2)
                    {
                        DigeoPoint const &ptSift2 = (*it2);
                        Pt3dr pTerrain = aCamera->F2AndZtoR3(Pt2dr(ptSift2.x,ptSift2.y),pt3.z);

                        double d2 = pow(pImg.x-ptSift2.x,2) + pow(pImg.y-ptSift2.y,2);
                        if (d2<(seuilPixel2))
                        {
                            double d = dist(ptSift,ptSift2);
                            if ((dmin<0)||(dmin>d))
                            {
                                dmin = d;
                                ptHomoDmin = ptSift2;
                                pTerrainDmin = pTerrain;
                            }
                        }
                    }
                    //std::cout << "dmin : "<<dmin<<" "<<ptHomoDmin.x<<" "<<ptHomoDmin.y<<std::endl;

                    if (dmin>=0)
                    {
                        // On valide le point avec la correlation
                        std::vector<double> fenImage;
                        std::vector<Pt2dr> vCoordImage;
                        double cmin = 0 ,cmax = 0 ,lmin = 0,lmax = 0;

                        for(int l=ptSift.y-SzW;l<=(ptSift.y+SzW);++l)
                        {
                            for(int c=ptSift.x-SzW;c<=(ptSift.x+SzW);++c)
                            {
                                Pt3dr P3D;
                                P3D.x = xminDalle + c * resolution;
                                P3D.y = ymaxDalle - l * resolution;
                                P3D.z = pTerrainDmin.z;

                                Pt2dr p2 = aCamera->R3toF2(P3D);
                                vCoordImage.push_back(p2);
                                if (vCoordImage.size()==1)
                                {
                                    cmin = p2.x;
                                    cmax = p2.x;
                                    lmin = p2.y;
                                    lmax = p2.y;
                                }
                                else
                                {
                                    if (cmin>p2.x)
                                        cmin=p2.x;
                                    else if (cmax<p2.x)
                                        cmax=p2.x;
                                    if (lmin>p2.y)
                                        lmin=p2.y;
                                    else if (lmax<p2.y)
                                        lmax=p2.y;
                                }
                            }
                        }
                        Pt2di PminCrop((int)round_ni(cmin-1.),(int)round_ni(lmin-1.));
                        Pt2di SzCrop((int)round_ni(cmax-PminCrop.x+2),(int)round_ni(lmax-PminCrop.y+2));
                        std::cout << "Crop : "<<PminCrop.x<<" "<<PminCrop.y<<" / "<<SzCrop.x<<" "<<SzCrop.y<<std::endl;
                        std_unique_ptr<TIm2D<U_INT2,INT4> > cropImg(createTIm2DFromFile<U_INT2,INT4>((*itLF),PminCrop,SzCrop));
                        if (cropImg.get()==NULLPTR)
                        {
                            cerr << "Error in "<<(*itLF)<<" Crop : "<<PminCrop.x<<" "<<PminCrop.y<<" / "<<SzCrop.x<<" "<<SzCrop.y<<std::endl;
                            return EXIT_FAILURE;
                        }
                        for(size_t i=0;i<vCoordImage.size();++i)
                        {
                            fenImage.push_back(cropImg->getr(Pt2dr(vCoordImage[i].x-PminCrop.x,vCoordImage[i].y-PminCrop.y),NoData));
                        }

                        // Coeff de correlation
                        double coef = correl(fenOrtho,fenImage,NoData);
                        std::cout << "coef : "<<coef<<" Point Ortho : "<<ptSift.x<<" "<<ptSift.y<<" Point Img : "<<ptHomoDmin.x<<" "<<ptHomoDmin.y<<std::endl;
                        if (coef >= seuilCorrel)
                        {
                            std::cout << "Point Dmin : "<<ptHomoDmin.x<<" "<<ptHomoDmin.y<<" Correl="<<coef<<std::endl;
                            ficAppuis << idAmer <<" "<<numImg<<" "<<ptHomoDmin.y<<" "<<ptHomoDmin.x<<" 5.00e-01  5.00e-01 0"<<std::endl;
                            //TODO: a verifier greg: il manque usePt =true; ?
                            //vraiment jamais utilisé a cause du #if ?

                            if (aExportMM)
                            {
                                cOneMesureAF1I aMesure;

                                aMesure.PtIm()=ptMaxCorrel;

                                std::ostringstream oss;
                                oss << idAmer;
                                aMesure.NamePt()=oss.str();

                                aDicoMesure.OneMesureAF1I().push_back(aMesure);
                            }
                        }
                    }
#endif

                    ++itCamera;
                    ++itPoi;
                    ++numImg;

                    if (aExportMM && usePt) aSetDicoMesure.MesureAppuiFlottant1Im().push_back(aDicoMesure);
                }
                if (usePt)
                {
                    ficAmers << std::fixed << idAmer<<" "<< pt3.x<<" "<<pt3.y<<" "<<pt3.z<<" 0.0 0.0 0.0 0.1 0.1 0.1 1"<<std::endl;

                    if (aExportMM)
                    {
                        cOneAppuisDAF aOAD;
                        aOAD.Pt() = pt3;
                        std::ostringstream oss;
                        oss << idAmer;
                        aOAD.NamePt() = oss.str();
                        aOAD.Incertitude() = Pt3dr(1,1,1);

                        aDicoAppuis.OneAppuisDAF().push_back(aOAD);
                    }

                    ++idAmer;
                }
            }

            if (aExportMM)
            {
                std::string aBaseNameResult = "measures"; //TODO: mettre en arg optionnel

                MakeFileXML(aDicoAppuis,aBaseNameResult+"-S3D.xml");
                MakeFileXML(aSetDicoMesure,aBaseNameResult+"-S2D.xml");
            }
        }
    }

    /*
    // Extraction des POI
    {
        std::string cmdPOI="mm3d Digeo ortho.tif -o ortho.dat";
        system(cmdPOI.c_str());
    }
    // Extraction du MNT
    {
        std::ostringstream oss;
        oss << std::fixed << "curl -o mnt.bil -H='Referer: http://localhost' \"http://wxs-i.ign.fr/"<<aKeyGPP<<"/geoportail/r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=ELEVATION.ELEVATIONGRIDCOVERAGE&STYLES=normal&FORMAT=image/x-bil;bits=32&BBOX="<< xminChantier<<","<<yminChantier<<","<<xmaxChantier<<","<<ymaxChantier<<"&CRS=EPSG:2154&WIDTH="<<NC<<"&HEIGHT="<<NL<<"\"";
        std::cout << "commande : "<<oss.str()<<std::endl;
        system(oss.str().c_str());


        //echo 'NROWS 675\nNCOLS 1769\nNBANDS 1\nBYTEORDER I\nNBITS 32\nLAYOUT  BIL\nSIGNE 1\nBAND_NAMES Z\n' > mnt.HDR
        std::ostringstream ossHdr;
        ossHdr << "echo 'NROWS "<<NL<<"\nNCOLS "<<NC<<"\nNBANDS 1\"nBYTEORDER I\nNBITS 32\nLAYOUT  BIL\nSIGNE 1\nBAND_NAMES Z\n' > mnt.HDR";
        system(ossHdr.str().c_str());
    }
     */

    /*
    // Il faut convertir cette emprise en coordonnees geographique
    //    command = "cs2cs "+targetSyst+" +to +proj=latlon +datum=WGS84 +ellps=WGS84 -f %.12f -s  processing/indirect_ptCarto.txt >  processing/indirect_ptGeo.txt";
    //echo 368225.296078 6555181.166326 | cs2cs +init=IGNF:LAMB93 +to +proj=latlon +datum=WGS84
    //echo 368225.296078 6555181.166326 > tempCarto.txt
    //cs2cs +init=IGNF:LAMB93 +to +proj=latlon +datum=WGS84 -f %.12f tempCarto.txt > tempGeo.txt
    {
        std::ofstream ficTempCarto("tempCarto.txt");
        ficTempCarto << std::fixed << xminChantier<<" "<<yminChantier<<std::endl;
        ficTempCarto << std::fixed << xminChantier<<" "<<ymaxChantier<<std::endl;
        ficTempCarto << std::fixed << xmaxChantier<<" "<<ymaxChantier<<std::endl;
        ficTempCarto << std::fixed << xmaxChantier<<" "<<yminChantier<<std::endl;
    }
    std::string proj4cmd("cs2cs +init=IGNF:LAMB93 +to +proj=latlon +datum=WGS84 -f %.12f tempCarto.txt > tempGeo.txt");
    command(proj4cmd.c_str());
    double latmax,lonmin;
    {
        bool first = true;
        std::iftream ficTempGeo("tempGeo.txt");
        double lon,lat;
        ficTempGeo >> lon >> lat;
        if (fic.good())
        {
            if (first)
            {
                first = false;
                latmax = lat;
                lonmin = lon;
            }
            else
            {
                if (lat>latmax)
                    latmax=lat;
                if (lon<lonmin)
                    lonmin=lon;
            }
        }
    }
    std::cout << "Emprise Geo : "<<latmax<<" "<<lonmin<<std::endl;
    */

    // On extrait les POI de l'ortho

    // Export des fichiers
    // PtAppuis.txt (NumPt,lon,lat,alti)
    // Appuis.txt (NumPt,NumImage,ligne,colonne)
    // PtLiaisons.txt (NumPt,alti)
    // Liaisons.txt (NumPt,NumImage,ligne,colonne)


    return 0;
}

#ifdef WIN32
#else
    #ifndef __APPLE__
        #pragma GCC diagnostic pop
    #endif
#endif


