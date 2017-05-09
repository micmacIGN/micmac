#include "Sat.h"
#include "../uti_phgrm/MICMAC/cCameraModuleOrientation.h"
#include "../uti_phgrm/MICMAC/Jp2ImageLoader.h"
#include "../uti_phgrm/MICMAC/cInterfModuleImageLoader.h"

#ifdef WIN32
#else
    #ifndef __APPLE__
        #pragma GCC diagnostic push
    #endif
    #pragma GCC diagnostic ignored "-Wunused-result"
#endif

/*{
 Pt2di ptImage_NO (int(ptImage.x),int(ptImage.y));
 if (getCorelCoef(ptImage_NO,orthoSz,aNameImage,fenOrtho,NoData)>coefOk)
 {
 mPix.insert(std::pair<Pt3dr,Pt2dr>(itGCP->first,ptImage));
 break;
 }
 Pt2di ptImage_NE (int(ptImage.x),int(ptImage.y+1));
 if (getCorelCoef(ptImage_NE,orthoSz,aNameImage,fenOrtho,NoData)>coefOk)
 {
 mPix.insert(std::pair<Pt3dr,Pt2dr>(itGCP->first,ptImage));
 break;
 }
 Pt2di ptImage_SO (int(ptImage.x+1),int(ptImage.y));
 if (getCorelCoef(ptImage_SO,orthoSz,aNameImage,fenOrtho,NoData)>coefOk)
 {
 mPix.insert(std::pair<Pt3dr,Pt2dr>(itGCP->first,ptImage));
 break;
 }
 Pt2di ptImage_SE (int(ptImage.x+1),int(ptImage.y+1));
 if (getCorelCoef(ptImage_SE,orthoSz,aNameImage,fenOrtho,NoData)>coefOk)
 {
 mPix.insert(std::pair<Pt3dr,Pt2dr>(itGCP->first,ptImage));
 break;
 }
 }*/

///
///
///
double correlVec(std::vector<double> const &f1,std::vector<double> const &f2)
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
    double pdtVar = (sa2-pow(sa,2)/N)*(sb2-pow(sb,2)/N);
    if (pdtVar!=0.)
        return (sab-sa*sb/N)/sqrt(pdtVar);
    return 0.;
}

///
///
///
std::string getGPPAccess(const std::string aHttpProxy)
{
    std::ostringstream oss;
    oss << g_externalToolHandler.get( "curl" ).callName() << " -H='Referer: http://localhost' ";
    if (!aHttpProxy.empty())
        oss << "-x "<<aHttpProxy;
    return oss.str();
}


///
///
///
void getVignetteGPP(const std::string nomDalleOrtho, const std::string aKeyGPP, const std::string aHttpProxy,
                    const Pt3dr& ptSol, const size_t sizeCropBDOrtho,
                    const float resolBDOrtho, const float resolImageSat)
{
    std::string gppAccess = getGPPAccess(aHttpProxy);
    
    //dalle
    double xminDalle = ptSol.x - sizeCropBDOrtho*resolBDOrtho;
    double yminDalle = ptSol.y - sizeCropBDOrtho*resolBDOrtho;
    double xmaxDalle = ptSol.x + sizeCropBDOrtho*resolBDOrtho;
    double ymaxDalle = ptSol.y + sizeCropBDOrtho*resolBDOrtho;
    
    //taille de la dalle
    int ncDalle = 2*sizeCropBDOrtho*resolBDOrtho/resolImageSat;
    int nlDalle = 2*sizeCropBDOrtho*resolBDOrtho/resolImageSat;
    
    //recuperation de l'ortho
    std::ostringstream ossGPP;
    ossGPP << std::fixed << gppAccess << " -o "<<nomDalleOrtho<< " \"http://wxs-i.ign.fr/"<<aKeyGPP<<"/geoportail/r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=ORTHOIMAGERY.ORTHOPHOTOS&STYLES=normal&FORMAT=image/geotiff&BBOX="<< xminDalle<<","<<yminDalle<<","<<xmaxDalle<<","<<ymaxDalle<<"&CRS=EPSG:2154&WIDTH="<<ncDalle<<"&HEIGHT="<<nlDalle<<"\"";
    
    std::cout << ossGPP.str() << std::endl;
    system(ossGPP.str().c_str());
}

///
///
///
std_unique_ptr<TIm2D<U_INT2,INT4> > getTIm2DFromImage(const std::string nomImage,
                                                      const int cmin, const int lmin,
                                                      const int sizeX, const int sizeY)
{
    
    Pt2di PminCrop(cmin,lmin);
    Pt2di SzCrop(sizeX,sizeY);
    std_unique_ptr<TIm2D<U_INT2,INT4> > cropImg(createTIm2DFromFile<U_INT2,INT4>(nomImage,PminCrop,SzCrop));
    return cropImg;
}

///
///
///
void getVecFromTIm2D(const std_unique_ptr<TIm2D<U_INT2,INT4> >& cropImg, std::vector<double>& fenImage,
                     const double NoData)
{
    for (size_t l(0); l<cropImg->sz().y; l++)
    {
        for (size_t c(0); c<cropImg->sz().x; c++)
        {
            fenImage.push_back(cropImg->getr(Pt2dr(c,l),NoData));
        }
    }
}

///
///
///
double getCorelCoef(const Pt2di& ptImage, const Pt2di& orthoSz, const std::string nomImage,
                    const std::vector<double> fenRef, const double NoData)
{
    std::vector<double> fenImage_i;
    int cmin = ptImage.x - orthoSz.x/2;
    int lmin = ptImage.y - orthoSz.y/2 ;
    std_unique_ptr<TIm2D<U_INT2,INT4> > cropImg = getTIm2DFromImage(nomImage,cmin,lmin,orthoSz.y,orthoSz.x);
    getVecFromTIm2D(cropImg,fenImage_i,NoData);
    
    //correlation :
    return correlVec(fenRef,fenImage_i);
}

///
///
///
Pt2di ImageSize(std::string const &aName)
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

///
///
///
bool getPoints(const std::string aBDCFileName, std::map<size_t,Pt3dr>& mGCP)
{
    std::ifstream fic (aBDCFileName);
    if (!fic.good())
    {
        cerr << "Error in reading "<<aBDCFileName<< " !" << std::endl;
        return false;
    }
    
    std::string line;
    while ( std::getline(fic, line ) )
    {
        Pt3dr Pt; size_t nPt(0);
        std::istringstream iss4(line);
        iss4 >> nPt >> Pt.x >> Pt.y >> Pt.z;
        mGCP.insert(std::pair<size_t,Pt3dr>(nPt,Pt));
    }
    
    return true;
}

///
///
///
int GCP_From_BDCarrefour_main(int argc, char **argv)
{
    bool verbose  (1);
    bool verbose2 (1);
    
    if (verbose) std::cout << "[GCP_From_BDCarrefour_main] Get BDOrtho control point for bundle" << std::endl;
    
    std::string aBDCFileName;
    std::string aOutDir;
    std::string aNameImage,aNameImageOri;
    
    //pour connexion GPP :
    std::string aKeyGPP;
    std::string aHttpProxy;
    
    ElInitArgMain
    (
         argc, argv,
         LArgMain() << EAMC(aBDCFileName,"Name of the GCP file")
                    << EAMC(aNameImage,"Name of the image")
                    << EAMC(aNameImageOri,"Name of the image orientation")
                    << EAMC(aOutDir,"Name of output directory")
                    << EAMC(aKeyGPP,"GPP Key"),
        LArgMain()  << EAM(aHttpProxy,"Proxy",true,"http proxy for GPP access")
    );
    
    //lecture des points de contrôle
    std::map<size_t,Pt3dr> mGCP;
    if (!getPoints(aBDCFileName,mGCP)) return EXIT_FAILURE;
    
   if (verbose) std::cout << "[GCP_From_BDCarrefour_main] Nb of points : " << mGCP.size() << std::endl;

    // chargement de l'orientation de l'image sat
    ElAffin2D oriIntImaM2C;
    Pt2di ImgSz = ImageSize(aNameImage);
    std_unique_ptr<ElCamera> aCam (new cCameraModuleOrientation(new OrientationGrille(aNameImageOri),ImgSz,oriIntImaM2C));
   if (verbose) std::cout << "[GCP_From_BDCarrefour_main] input image resolution : "<< aCam->ResolutionSol() << std::endl;

    float resolImageSat=aCam->ResolutionSol();//todo : le sol doit etre a 0. : a voir si ce n'est pas le cas.
    float resolBDOrtho = 0.20;//connu a priori
    
    size_t sizeCropBDOrtho = 50;
    double NoData = -1.;
    int marge = 10;
    double coefOk=0.9;
    
    //  lecture de la BDOrtho et recuperation d'un crop
    //  on trouve le point par correlation
    if (verbose) std::cout << "[GCP_From_BDCarrefour_main] for each points..." << std::endl;
    std::map<size_t,Pt3dr>::iterator itGCP (mGCP.begin()), endGCP (mGCP.end());
    std::map<size_t,Pt2dr> mPix; std::map<size_t,Pt2dr>::iterator itPix;
    for (;itGCP!=endGCP;itGCP++)
    {
        if (verbose2) std::cout << "[GCP_From_BDCarrefour_main] point " << itGCP->first << std::endl;
        
        //recuperation de la vignette
        std::ostringstream ossNomOrtho;
        ossNomOrtho << "Ortho_"<<itGCP->first<<".tif";
        std::string nomDalleOrtho = ossNomOrtho.str();
        if (verbose2) std::cout << "[GCP_From_BDCarrefour_main] point " << itGCP->first << " : vignette" << std::endl;
        getVignetteGPP(nomDalleOrtho,aKeyGPP,aHttpProxy,itGCP->second,sizeCropBDOrtho,resolBDOrtho,resolImageSat);
        
        //info de l'image :
        int ncDalle = 2*sizeCropBDOrtho*resolBDOrtho/resolImageSat;
        int nlDalle = 2*sizeCropBDOrtho*resolBDOrtho/resolImageSat;
        Pt2di orthoSz (ncDalle,nlDalle);
        if (verbose2) std::cout << "[GCP_From_BDCarrefour_main] point " << orthoSz.x << " " << orthoSz.y << std::endl;
        
        //BDOrtho => tab
        std::vector<double> fenOrtho;
        std_unique_ptr<TIm2D<U_INT2,INT4> > cropBDOrtho = getTIm2DFromImage(nomDalleOrtho,0,0,orthoSz.y,orthoSz.x);
        getVecFromTIm2D(cropBDOrtho,fenOrtho,NoData);
        
        //point dans l'image :
        Pt2dr ptImage = aCam->Ter2Capteur(itGCP->second);
        if (verbose2) std::cout << "[GCP_From_BDCarrefour_main] point in input image " << ptImage.x << " " << ptImage.y << std::endl;
        double CoefCorelMax(-1);

        //on regarde si le pt image est bon :
        //si la corel marche bien sur un pixel autour du point de loc, dans le doute on garde la loc...
        //todo : faire mieux...
     
        
        //correlations :
        for (int i(0); i<marge;i++)
        {
            Pt2di ptImage_i (int(ptImage.x)-marge+i,int(ptImage.y)-marge+i);
            double cofCorel = getCorelCoef(ptImage_i,orthoSz,aNameImage,fenOrtho,NoData);
            
            if(cofCorel>CoefCorelMax)
            {
                CoefCorelMax = cofCorel;
                if (verbose2) std::cout << "[GCP_From_BDCarrefour_main] " << itGCP->first << " : " << cofCorel << "  pt (" << ptImage_i.x <<"," <<ptImage_i.y<<")" << std::endl;
                
                itPix = mPix.find(itGCP->first);
                Pt2dr ptr (ptImage_i.x,ptImage_i.y);
                if (itPix==mPix.end())
                    mPix.insert(std::pair<size_t,Pt2dr>(itGCP->first,ptr));
                else
                    itPix->second = ptr;
            }
        }
        
        //sortie de la meilleure image :
        {
            std::ostringstream ossNomVignetteResult;
            ossNomVignetteResult << "Vignette_"<<itGCP->first<<".tif";
            std::string nomVignetteResult = ossNomVignetteResult.str();

            itPix = mPix.find(itGCP->first);
            if (itPix == mPix.end()) return EXIT_FAILURE;
            
            if (verbose2) std::cout << "[GCP_From_BDCarrefour_main] (" << ptImage.x << ";" << ptImage.y << ") => (" << itPix->second.x << ";" << itPix->second.y << ")" << std::endl;
            
            int cmin = itPix->second.x - orthoSz.x/2;
            int lmin = itPix->second.y - orthoSz.y/2 ;
            std_unique_ptr<TIm2D<U_INT2,INT4> > cropImg = getTIm2DFromImage(aNameImage,cmin,lmin,orthoSz.y,orthoSz.x);
            
            Tiff_Im out(nomVignetteResult.c_str(), cropImg->sz(),GenIm::u_int2,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);
            ELISE_COPY(cropImg->_the_im.all_pts(),cropImg->_the_im.in(),out.out());
        }
        
    }
    
    return 0;
}

#ifdef WIN32
#else
    #ifndef __APPLE__
        #pragma GCC diagnostic pop
    #endif
#endif


