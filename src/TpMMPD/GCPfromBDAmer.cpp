#include "Sat.h"
#include "../uti_phgrm/MICMAC/cCameraModuleOrientation.h"
#include "../uti_phgrm/MICMAC/cInterfModuleImageLoader.h"
#include "../uti_phgrm/MICMAC/Jp2ImageLoader.h"
#include "GCPfromBDAmer.h"

#ifdef WIN32
#else
    #ifndef __APPLE__
        #pragma GCC diagnostic push
    #endif
    #pragma GCC diagnostic ignored "-Wunused-result"
#endif

///
///
///
int getPlaceLastChar(const std::string fileName, const char car)
{
    int place = -1;
    for(int l=(int)(fileName.size()-1);(l>=0)&&(place==-1);--l)
    {
        if (fileName[l]==car)
        {
            place = l;
        }
    }
    return place;

}

///
///
///
int getPlacePoint(const std::string fileName)
{
    return getPlaceLastChar(fileName,'.');
}

///
///
///
int getPlaceLastSlash(const std::string fileName)
{
    int place = getPlaceLastChar(fileName,'/');
    if (place==-1)
        place = getPlaceLastChar(fileName,'\\');
    return place;
}

///
///
///
std::string getFileName(const std::string fileName)
{
    int placeSlah = getPlaceLastSlash(fileName);
    int placePoint = getPlacePoint(fileName);
    
    std::string name("");
    for(int l=placeSlah;l<placePoint;l++)
        name += fileName[l];
    
    return name;
}

///
///
///
std::string getFileDir(const std::string fileName)
{
    int placeSlah = getPlaceLastSlash(fileName);
    
    std::string name("");
    for(int l=0;l<placeSlah;l++)
        name += fileName[l];
    
    return name;
}

///
///
///
double minVec(std::vector<double>& c)
{
    double ret = c[0];
    for (size_t i(1); i<c.size(); i++)
        if (c[i]<ret)
            ret = c[i];
    return ret;
}
double maxVec(std::vector<double>& c)
{
    double ret = c[0];
    for (size_t i(1); i<c.size(); i++)
        if (c[i]>ret)
            ret = c[i];
    return ret;
}


/////
namespace GCPfromBDI_files
{
    
    ///
    ///
    ///
    std::list<std::string> getFiles(const std::string fileNameWithPattern)
    {
        std::string aDir,aPat;
        SplitDirAndFile(aDir,aPat,fileNameWithPattern);
        cInterfChantierNameManipulateur *aICNM;
        aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
        std::list<std::string> listFinal, list = aICNM->StdGetListOfFile(aPat);
        
        std::list<std::string>::const_iterator it (list.begin()),fin=list.end();
        for(;it!=fin;++it)
            listFinal.push_back(aDir+"/"+*it);
        
        return listFinal;
    }
    
}

/////
namespace GCPfromBDI_Image
{
    ///
    ///
    ///
    bool getOri(const std::string imageName, const std::string aGRIDExt, std::string& orientationName)
    {
        int placePoint = getPlacePoint(imageName);
        if (placePoint==-1) return false;
        
        std::string baseName;
        baseName.assign(imageName.begin(),imageName.begin()+placePoint+1);
        
        orientationName = baseName+aGRIDExt;
        return (ELISE_fp::exist_file(orientationName));
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
    bool calcPseudoOrtho(const std::string& aNameImage, const Pt3dr& PtNO, const Pt2di& size,
                         const float resol, const std_unique_ptr<ElCamera>& aCam,
                         std_unique_ptr<TIm2D<U_INT2,INT4> >& vignette)
    {
        
        Pt3dr PtNE (PtNO.x+size.x*resol,    PtNO.y,                 PtNO.z);
        Pt3dr PtSO (PtNO.x,                 PtNO.y-size.y*resol,    PtNO.z);
        Pt3dr PtSE (PtNO.x+size.x*resol,    PtNO.y-size.y*resol,    PtNO.z);
        
        Pt2dr ptINO = aCam->Ter2Capteur(PtNO);
        Pt2dr ptINE = aCam->Ter2Capteur(PtNE);
        Pt2dr ptISO = aCam->Ter2Capteur(PtSO);
        Pt2dr ptISE = aCam->Ter2Capteur(PtSE);
        
        std::vector<double> vc; vc.push_back(ptINO.x); vc.push_back(ptINE.x); vc.push_back(ptISO.x); vc.push_back(ptISE.x);
        std::vector<double> vl; vl.push_back(ptINO.y); vl.push_back(ptINE.y); vl.push_back(ptISO.y); vl.push_back(ptISE.y);
        
        int cmin = floor(minVec(vc));
        int cmax = floor(maxVec(vc))+1;
        int lmin = floor(minVec(vl));
        int lmax = floor(maxVec(vl))+1;
        
        int buffer (1);
        Pt2di minIm  (cmin-buffer,lmin-buffer);
        if (minIm.x<0 || minIm.y<0) return false;
        
        Pt2di sizeImSrc = GCPfromBDI_Image::ImageSize(aNameImage);
        Pt2di sizeIm (cmax-cmin+2*buffer,lmax-lmin+2*buffer);
        
        if ( minIm.x + sizeIm.x >= sizeImSrc.x || minIm.y + sizeIm.y >=sizeImSrc.y ) return false;
        
        std_unique_ptr<TIm2D<U_INT2,INT4> > cropImg_cl (createTIm2DFromFile<U_INT2,INT4>(aNameImage,minIm,sizeIm));
        
        for (size_t l(0); l<size.y ; l++)
        {
            for (size_t c(0); c<size.x ; c++)
            {
                Pt3dr pt (PtNO.x + c*resol, PtNO.y - l*resol, PtNO.z);
                Pt2dr ptImage = aCam->Ter2Capteur(pt);
                Pt2dr ptImageCrop (ptImage.x-minIm.x,ptImage.y-minIm.y);
                
                double radio = cropImg_cl->getr(ptImageCrop);
                vignette->oset(Pt2di(c,l),(int)radio);
            }
        }
        
        return true;
    }

    
}

/////
namespace GCPfromBDI_Export
{
    
    ///
    ///
    ///
    void WriteGCP3D(const std::map<std::string,Pt3dr>& mGCP, const std::string nomFic)
    {
        std::ofstream fic (nomFic.c_str());
        fic << std::fixed << std::setprecision(5);
        fic << "<?xml version=\"1.0\" ?>" << std::endl;
        fic << "<DicoAppuisFlottant>" << std::endl;
        
        std::map<std::string,Pt3dr>::const_iterator itGCP (mGCP.begin()), endGCP (mGCP.end());
        for (;itGCP!=endGCP;itGCP++)
        {
            fic << "    <OneAppuisDAF>" << std::endl;
            fic << "        <Pt>" << itGCP->second.x << " " << itGCP->second.y << " " << itGCP->second.z << "</Pt>" << std::endl;
            fic << "        <NamePt>" << itGCP->first << "</NamePt>" << std::endl;
            fic << "        <Incertitude>" << "1 1 1" << "</Incertitude>" << std::endl;//pour le moment on ne joue pas avec les incertitudes
            fic << "    </OneAppuisDAF>" << std::endl;
        }
        
        fic << "</DicoAppuisFlottant>" << std::endl;
        fic.close();
    }
    
    
    ///
    ///
    ///
    void WriteGCP2D(const MPtImage& mptImage, const std::string nomFic)
    {
        std::ofstream fic (nomFic.c_str());
        fic << std::fixed << std::setprecision(5);
        fic << "<?xml version=\"1.0\" ?>" << std::endl;
        fic << "<SetOfMesureAppuisFlottants>" << std::endl;
        
        ItPtImage itImage (mptImage.begin()), endImage (mptImage.end());
        for (;itImage!=endImage;itImage++)
        {
            fic << "    <MesureAppuiFlottant1Im>"<<std::endl;
            fic << "        <NameIm>" << itImage->first <<"</NameIm>"<<std::endl;
            
            const MPtCoordPix& mptCoord = itImage->second;
            cItPtCoordPix itPt (mptCoord.begin()), endPt (mptCoord.end());
            for (;itPt!=endPt;itPt++)
            {
                fic << "        <OneMesureAF1I>"<<std::endl;
                fic << "            <NamePt>" << itPt->first << "</NamePt>"<<std::endl;
                fic << "            <PtIm>" << itPt->second.x << " " << itPt->second.y << "</PtIm>"<<std::endl;
                fic << "        </OneMesureAF1I>"<<std::endl;
            }
            
            fic << "    </MesureAppuiFlottant1Im>"<<std::endl;
        }
        
        fic << "</SetOfMesureAppuisFlottants>" << std::endl;
        fic.close();
    }
}

/////
namespace GCPfromBDI_Correl
{

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
        std_unique_ptr<TIm2D<U_INT2,INT4> > cropImg (createTIm2DFromFile<U_INT2,INT4>(nomImage,ptImage,orthoSz));
        getVecFromTIm2D(cropImg,fenImage_i,NoData);
        
        //correlation :
        return correlVec(fenRef,fenImage_i);
    }

}

////
namespace BDAmer
{
    ///
    ///
    ///
    BASE_AMERS::BASE_AMERS(const std::string& fileName , const std::string nomBase):
    nom(nomBase)
    {
        this->read(fileName);
    }
    
    ///
    ///
    ///
    bool BASE_AMERS::read(const std::string& fileName)
    {
        cElXMLTree aTrBDA(fileName);
        
        cElXMLTree * aTrBaseAmer = aTrBDA.GetUnique("BASE_AMERS");
        if (aTrBaseAmer == NULL) return false;
        
        //auteur :
        cElXMLTree * aTrAuteur = aTrBaseAmer->GetUnique("auteur");
        if (aTrAuteur != NULL) this->auteur = aTrAuteur->GetUniqueVal();
        
        //date :
        cElXMLTree * aTrDate = aTrBaseAmer->GetUnique("date");
        if (aTrDate != NULL) this->date = aTrDate->GetUniqueVal();

        //origine :
        cElXMLTree * aTrOrigine = aTrBaseAmer->GetUnique("origine");
        if (aTrOrigine != NULL)
        {
        }

        //filtrages :
        cElXMLTree * aTrFiltrages = aTrBaseAmer->GetUnique("filtrages");
        if (aTrFiltrages != NULL)
        {
        }

        //dalles :
        std::list<cElXMLTree*> lDalles = aTrBaseAmer->GetAll("dalle");
        std::list<cElXMLTree*>::const_iterator iDalle;
        for( iDalle=lDalles.begin(); iDalle!=lDalles.end(); iDalle++ )
        {
            std::unique_ptr<Dalle> dalle (new Dalle);
            if (!dalle->read(*iDalle))
                return false;
            dalle->base = this;
            vDalles.push_back(*dalle.release());
        }
        
        return true;
    }
    
    ///
    ///
    ///
    void BASE_AMERS::getValidPoint(const Box2dr& box, std::vector<Point>& vPt)
    {
        for (size_t i(0); i<vDalles.size(); i++)
            vDalles[i].getValidPoint(box,vPt);
    }
    
    ///
    ///
    ///
    bool Dalle::read(cElXMLTree* aTrDalle)
    {
        //nom :
        cElXMLTree * aTrNom= aTrDalle->Get("nom");
        this->nom = aTrNom->GetUniqueVal();
        std::cout << "nom de la dalle : " << this->nom << std::endl;
        
        //nombre de points :
        cElXMLTree * aTrNbPoints= aTrDalle->Get("nombre_de_points");
        if (aTrNbPoints == NULL) return false;
        {
            cElXMLTree * aTrTotal = aTrNbPoints->Get("total");
            this->nombre_de_points_total = aTrTotal->GetUniqueValInt();
            
            cElXMLTree * aTrValid = aTrNbPoints->Get("valides");
            this->nombre_de_points_valid = aTrValid->GetUniqueValInt();
            if (this->nombre_de_points_valid==0) return false;
        }
        
        //archive_imagettes :
        cElXMLTree * aTrArchImage= aTrDalle->Get("archive_imagettes");
        if (aTrArchImage != NULL) this->archive_imagettes = aTrArchImage->GetUniqueVal();
        else return false;

        //chargement des sous dalles :
        std::list<cElXMLTree*> lSousDalles = aTrDalle->GetAll("sous_dalle");
        std::list<cElXMLTree*>::const_iterator iSousDalle;
        for( iSousDalle=lSousDalles.begin(); iSousDalle!=lSousDalles.end(); iSousDalle++ )
        {
            std::unique_ptr<SousDalle> sousdalle (new SousDalle);
            if (!sousdalle->read(*iSousDalle))
                return false;
            sousdalle->dalle = this;
            vSousDalles.push_back(*sousdalle.release());
        }

        return true;
    }
    
    ///
    ///
    ///
    void Dalle::getValidPoint(const Box2dr& box, std::vector<Point>& vPt)
    {
        for (size_t i(0); i<vSousDalles.size(); i++)
            vSousDalles[i].getValidPoint(box,vPt);
    }
    
    ///
    ///
    ///
    bool SousDalle::read(cElXMLTree* aTrSousDalle)
    {
        //nom :
        cElXMLTree * aTrNum = aTrSousDalle->Get("numero");
        this->numero = aTrNum->GetUniqueVal();
        
        //chargement des points :
        cElXMLTree * aTrPoints= aTrSousDalle->Get("points");
        std::list<cElXMLTree*> lPoints = aTrPoints->GetAll("point");
        std::list<cElXMLTree*>::const_iterator iPoints;
        for( iPoints=lPoints.begin(); iPoints!=lPoints.end(); iPoints++ )
        {
            std::unique_ptr<Point> point (new Point);
            if (!point->read(*iPoints))
                return false;
            point->sousDalle = this;
            vPoints.push_back(*point.release());
            vPoints[vPoints.size()-1].sousDalle = this;
        }

        return true;
    }
    
    ///
    ///
    ///
    void SousDalle::getValidPoint(const Box2dr& box, std::vector<Point>& vPt)
    {
        for (size_t i(0); i<vPoints.size(); i++)
        {
            if (!vPoints[i].isValid()) continue;
            if (box.Include(vPoints[i].coordTer))
                vPt.push_back(vPoints[i]);
        }
    }
    
    ///
    ///
    ///
    bool Point::isValid()
    {
        for (size_t i(0); i<vSources.size(); i++)
        {
            if (vSources[i].validite) return true;
        }
        return false;
    }
    
    ///
    ///
    ///
    Pt3dr Point::getCoordTerrain() const
    {
        Pt3dr point (coordTer.x,coordTer.y,0.);
        
        point.z = hauteur_ellipsoide;

        return point;
    }
    
    
    ///
    ///
    ///
    std::string Point::getOrthoDir(const std::string BDAmerDir) const
    {
        const BDAmer::SousDalle* sousDalle = this->sousDalle;
        const BDAmer::Dalle* dalle = sousDalle->dalle;
        const BDAmer::BASE_AMERS* base = dalle->base;
        return  BDAmerDir + "/" + base->nom + "/" + dalle->nom + "/" + dalle->nom  + "_" + sousDalle->numero;
    }
    
    ///
    ///
    ///
    bool Point::read(cElXMLTree* aTrPoint)
    {
        //nom :
        cElXMLTree * aTrNom = aTrPoint->Get("nom");
        this->nom = aTrNom->GetUniqueVal();
        
        //coordonnees :
        cElXMLTree * aTrCoord = aTrPoint->Get("coordonnees");
        if (aTrCoord)
        {
            cElXMLTree * aTrLong = aTrPoint->Get("longitude");
            double Long = aTrLong->GetUniqueValDouble();
            
            cElXMLTree * aTrLat = aTrPoint->Get("latitude");
            double Lat = aTrLat->GetUniqueValDouble();

            this->coordTer = Pt2dr(Long,Lat);
            
            cElXMLTree * aTrH = aTrPoint->Get("hauteur_ellipsoide");
            if (aTrH != NULL) this->hauteur_ellipsoide = aTrH->GetUniqueValDouble();

            cElXMLTree * aTrAlEGM = aTrPoint->Get("altitude_egm96");
            if (aTrAlEGM != NULL) this->altitude_egm96 = aTrAlEGM->GetUniqueValDouble();

            cElXMLTree * aTrALRef = aTrPoint->Get("altitude_ref3d");
            if (aTrALRef != NULL) this->altitude_ref3d = aTrALRef->GetUniqueValDouble();

            cElXMLTree * aTrSRTM = aTrPoint->Get("altitude_srtm_dted1");
            if (aTrSRTM != NULL) this->altitude_srtm_dted1 = aTrSRTM->GetUniqueValDouble();
        }
        else return false;
        
        //qualite :
        cElXMLTree * aTrQualite = aTrPoint->Get("qualite");
        if (aTrQualite)
        {
            //rien pour le moment
        }

        //sources :
        cElXMLTree * aTrSource = aTrPoint->Get("sources");
        if (aTrSource)
        {
            //chargement des sous dalles :
            std::list<cElXMLTree*> lSources = aTrSource->GetAll("image");
            std::list<cElXMLTree*>::const_iterator iSources;
            for( iSources=lSources.begin(); iSources!=lSources.end(); iSources++ )
            {
                Source source;
                if (!source.read(*iSources))
                    return false;
                vSources.push_back(source);
            }

        }
        else return false;
        
        //correlation :
        cElXMLTree * aTrCorrel = aTrPoint->GetUnique("correlations");
        if (aTrCorrel)
        {
            //rien pour le moment
        }
        
        return true;
    }

    ///
    ///
    ///
    bool Source::read(cElXMLTree* aTrSource)
    {
        //A24 :
        cElXMLTree * aTrA24 = aTrSource->Get("A24");
        if (aTrA24 != NULL) this->A24 = aTrA24->GetUniqueVal();

        //validite :
        cElXMLTree * aTrAValid = aTrSource->Get("validite");
        this->validite = (aTrAValid->GetUniqueValInt()==1);
        
        //coordImage :
        cElXMLTree * aTrLig = aTrSource->Get("ligne");
        double lig = aTrLig->GetUniqueValDouble();
        cElXMLTree * aTrCol = aTrSource->Get("colonne");
        double col = aTrCol->GetUniqueValDouble();
        this->coordImage = Pt2dr(col,lig);

        //ortho :
        cElXMLTree * aTrOrtho = aTrSource->Get("ortho");
        this->ortho = aTrOrtho->GetUniqueVal();

        return true;
    }
        
}

///
///
///
void getBDAmerDalle(const Pt3dr& pt, std::string& general, std::string& detail)
{
    if (pt.y>0) general="N";
    if (pt.y>0) detail="N";
    
    int Y  = std::floor(pt.y/10)*10;
    int YY = std::floor(pt.y);
    
    if (Y==0) general += "0";
    if (YY<10) detail += "0";
    
    general += std::to_string(Y);
    detail  += std::to_string(YY);
    
    double x;
    if (pt.x<180) { general+="E";  detail+="E"; x = pt.x;}
    if (pt.x>180) { detail+="W"; general+="W"; x = 180-pt.x;}

    int X = std::floor(pt.x/10)*10;
    int XX = std::floor(pt.x);
    
    if (X<10)   general += "0";
    if (X<100)  general += "0";
    if (XX<10)  detail  += "0";
    if (XX<100) detail  += "0";
    
    general += std::to_string(X);
    detail  += std::to_string(XX);
}

///
///
///
Pt2dr toPt2dr(const Pt3dr& pt)
{
    return Pt2dr(pt.x,pt.y);
}

///
///
///
void addPtTer(const std::vector<BDAmer::Point>& vPt, GCPfromBDI_Export::MPtCoordTer& mGCP)
{
    for (size_t i(0); i<vPt.size(); i++)
        mGCP.insert(std::pair<std::string,Pt3dr>(vPt[i].nom,vPt[i].getCoordTerrain()));
}

///
///
///
int GCPfromBDAmer_main(int argc, char **argv)
{
    bool verbose  (1);
    
    if (verbose) std::cout << "[GCPfromBDAmer_main] Get BDAmer control point for bundle" << std::endl;

    // EAMC
    std::string aBDAmerDir;
    std::string aOutDir;
    std::string ImageNames;
    
    // EAM
    std::string aGRIDExt("GRI");
    double      aZmoy(0.);
    int         sizeCorrelation(200);
    double      CorefMinAccept(0.5);
    
    // valeurs par defaut utiles
    int     marge = 10;
    double  NoData = -1.;
    
    ElInitArgMain
    (
         argc, argv,
         LArgMain() << EAMC(ImageNames,"Full Name (Dir+Pat)")
                    << EAMC(aBDAmerDir,"Full Name (Dir+Pat)")
                    << EAMC(aOutDir,"Name of output directory"),
         LArgMain() << EAM(aGRIDExt,"Grid","GRI","Orientation file extension")
                    << EAM(aGRIDExt,"ZMoy","ZMoy","Approch altitude")
                    << EAM(sizeCorrelation,"CorSw",false,"Size Correlation Windows")
                    << EAM(CorefMinAccept,"CoefMini",false,"Minimum Coefficient to be acceptated")
    );
    
    // vignette de correlation :
    Pt2di SzVignette (sizeCorrelation,sizeCorrelation);
    
    //structure de sortie :
    GCPfromBDI_Export::MPtCoordTer  mGCP;
    GCPfromBDI_Export::MPtImage     mPtImage;
    
    //pour chaque image :
    std::list<std::string> lImage = GCPfromBDI_files::getFiles(ImageNames);
    std::list<std::string>::const_iterator itImage(lImage.begin()),finImage(lImage.end());
    for (;itImage!=finImage;++itImage)
    {

        if (verbose) std::cout << "Gestion de l'image " << *itImage << std::endl;
        
        std::string aNameImageOri;
        if (!GCPfromBDI_Image::getOri(*itImage,aGRIDExt,aNameImageOri))
        {
            std::cout << "No orientation file for " << *itImage << " - skip " << std::endl;
            continue;
        }

        // taille de l'image :
        Pt2di sizeImSrc = GCPfromBDI_Image::ImageSize(*itImage);
        
        // chargement de l'orientation de l'image sat
        ElAffin2D oriIntImaM2C;
        Pt2di ImgSz = GCPfromBDI_Image::ImageSize(*itImage);
        std_unique_ptr<ElCamera> aCam (new cCameraModuleOrientation(new OrientationGrille(aNameImageOri),ImgSz,oriIntImaM2C));
        
        // recherche de la dalle BDAmer :
        std::vector<Pt3dr> vPtBoxTer;
        vPtBoxTer.push_back(aCam->ImEtProf2Terrain(Pt2dr(0.,0.),aZmoy));
        vPtBoxTer.push_back(aCam->ImEtProf2Terrain(Pt2dr(sizeImSrc.x,sizeImSrc.y),aZmoy));
        Box2dr box (toPt2dr(vPtBoxTer[0]),toPt2dr(vPtBoxTer[1]));
        box.dilate(0.1); // valeur a revoir ?
        
        // creation d'un repertoire pour les imagettes :
        std::string vignetteNameDir = aOutDir + "/" + getFileName(*itImage);
        ELISE_fp::MkDir(vignetteNameDir);
        
        //structure de sortie
        GCPfromBDI_Export::MPtCoordPix mPix;
        
        // on rempli le set des fichiers de point a charger.
        for (size_t i(0); i<vPtBoxTer.size(); i++)
        {
            std::string general(""),detail("");
            getBDAmerDalle(vPtBoxTer[i],general,detail);

            std::string ficBDAmer = aBDAmerDir + "/" + general + "/" + detail + ".xml";
            BDAmer::BASE_AMERS bdamer (ficBDAmer,general);
            
            std::vector<BDAmer::Point> vPt;
            bdamer.getValidPoint(box,vPt);
            if (verbose) std::cout << "Nb de point(s) de test : " << vPt.size() << std::endl;

            // ajout des points terrain :
            addPtTer(vPt,mGCP);
            
            // coordonnées images :
            for (size_t i(0); i<1/*vPt.size()*/; i++)
            {
                const BDAmer::Point& point = vPt[i];
                std::cout << "traitement du point " << point.nom << std::endl;

                //point dans l'image :
                Pt3dr PtTerrain         = point.getCoordTerrain();
                std::cout << "PtTerrain " << PtTerrain.x << " " << PtTerrain.y << " " << PtTerrain.z << std::endl;

                Pt2dr ptImageApproch    = aCam->Ter2Capteur(PtTerrain);
                std::cout << "point dans l'image " << ptImageApproch.x << " " << ptImageApproch.y << std::endl;
                
                //resolution de l'image au Pt :
                double resolImageSat = aCam->ResolutionSol(PtTerrain);
                
                //calcul d'une pseudoOrtho :
                Pt2di POSz (SzVignette.x+2*marge,SzVignette.y+2*marge);
                Pt3dr PNOpseudoOrtho (PtTerrain.x-POSz.x*resolImageSat/2,PtTerrain.y+POSz.y*resolImageSat/2,PtTerrain.z);
                std_unique_ptr<TIm2D<U_INT2,INT4> > pseudoOrtho (new TIm2D<U_INT2,INT4>(POSz));
                if (!GCPfromBDI_Image::calcPseudoOrtho(*itImage,PNOpseudoOrtho,POSz,resolImageSat,aCam,pseudoOrtho))
                    return false;
                
                //export de la pseudoOrtho
                std::ostringstream ossNomPseudoOrtho;
                ossNomPseudoOrtho << vignetteNameDir << "/PseudoOrtho_"<<point.nom<<".tif";
                std::string nomPseudoOrtho = ossNomPseudoOrtho.str();
                Tiff_Im out(nomPseudoOrtho.c_str(), pseudoOrtho->sz(),GenIm::u_int2,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);
                ELISE_COPY(pseudoOrtho->_the_im.all_pts(),pseudoOrtho->_the_im.in(),out.out());
                
                std::string dirOrthoTGZ     = point.getOrthoDir(aBDAmerDir)+"_TIF.tgz";
                std::string dirOrthoOut     = vignetteNameDir + "/" + getFileName(dirOrthoTGZ);
                std::string dirOrthoOutTGZ  = dirOrthoOut + ".tgz";
                
                // copie et detar
                std::cout << "dirOrthoOut " << dirOrthoOut << std::endl;
                if (!ELISE_fp::exist_file(dirOrthoOut))
                {
                    std::cout << "copie et detar du fichier " << std::endl;
                    ELISE_fp::copy_file(dirOrthoTGZ,dirOrthoOutTGZ,true);
                    std::stringstream ossTar;  ossTar << "tar -zxf " << dirOrthoOutTGZ << " -C " << vignetteNameDir;
                    system(ossTar.str().c_str());
                    ELISE_fp::RmFileIfExist(dirOrthoOutTGZ);
                }
                
                //pour chaque source :
                double BestCorel (-1); Pt2dr ptr(0.,0.);
                for (size_t s(0); s<point.vSources.size(); s++)
                {
                    //on cherche par correlation le point dans l'image en entree
                    //on remplie la structure de sortie

                    const BDAmer::Source& source = point.vSources[s];
                    
                    const std::string nomImagette = vignetteNameDir + "/" + source.ortho;
                    std::cout << "nom de l'imagette : " << nomImagette << std::endl;
                    
                    if (!ELISE_fp::exist_file(nomImagette)) continue; //un pb dans le tar...
                    
                    std::vector<double> fenImagette;
                    std_unique_ptr<TIm2D<U_INT2,INT4> > cropBDOrtho (createTIm2DFromFile<U_INT2,INT4>(nomImagette,Pt2di(0,0),SzVignette));
                    GCPfromBDI_Correl::getVecFromTIm2D(cropBDOrtho,fenImagette,NoData);
                    
                    //correlations :
                    double CoefCorelMax(-1);
                    GCPfromBDI_Export::ItPtCoordPix itPix;
                    for (size_t l(0); l<POSz.y-SzVignette.y;l++)
                    {
                        for (size_t c(0); c<POSz.x-SzVignette.x;c++)
                        {
                            double cofCorel =  GCPfromBDI_Correl::getCorelCoef(Pt2di(c,l),SzVignette,nomPseudoOrtho,fenImagette,NoData);
                            
                            if(cofCorel>=CoefCorelMax)
                            {
                                if (cofCorel<CorefMinAccept) // pas assez bon
                                    continue;
                                
                                if (cofCorel<BestCorel) // deja mieux
                                    continue;

                                Pt2di ptCentralPO (c+SzVignette.x/2,l+SzVignette.y/2);
                                Pt3dr ptCentralPOTer ( PNOpseudoOrtho.x + ptCentralPO.x*resolImageSat,
                                                       PNOpseudoOrtho.y - ptCentralPO.y*resolImageSat,
                                                       PNOpseudoOrtho.z);

                                Pt2dr ptImage_i = aCam->Ter2Capteur(ptCentralPOTer);
                                CoefCorelMax = cofCorel;
                                BestCorel = cofCorel;
                                
                                ptr = Pt2dr(ptImage_i.x,ptImage_i.y);
                            }
                        }
                    }

                    itPix = mPix.find(point.nom);
                    if (itPix == mPix.end()) continue; // on a pas trouve de coef de correlation suffisant...
                    
                    
                }
                
                if (BestCorel>-1)
                {
                    mPix.insert(std::pair<std::string,Pt2dr>(point.nom,ptr));
                    if (verbose) std::cout << "[GCP_From_BDCarrefour_main] " << point.nom << " : " << BestCorel << " : (" << ptImageApproch.x << ";" << ptImageApproch.y << ") => (" << ptr.x << ";" << ptr.y << ")" << std::endl;
                }
                
                //un peu de menage
                ELISE_fp::RmDir(vignetteNameDir);
                
            }
            
        }
        
        // ajout des points images :
        mPtImage.insert(GCPfromBDI_Export::pPtImage(*itImage,mPix));
        
    }

    //export des points :
    std::ostringstream ossNomPt3D;
    ossNomPt3D << aOutDir << "/gcp-3D.xml";
    GCPfromBDI_Export::WriteGCP3D(mGCP,ossNomPt3D.str());
    
    std::ostringstream ossNomPt2D;
    ossNomPt2D << aOutDir << "/gcp-2D.xml";
    GCPfromBDI_Export::WriteGCP2D(mPtImage,ossNomPt2D.str());

    
    return 0;
}

#ifdef WIN32
#else
    #ifndef __APPLE__
        #pragma GCC diagnostic pop
    #endif
#endif


