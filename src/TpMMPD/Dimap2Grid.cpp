#include "StdAfx.h"

#if (ELISE_QT_VERSION >= 4)
#ifdef Int
#undef Int
#endif

#include <QFile>
#include <QDebug>
#include <QXmlStreamReader>

class Dimap{
public:
    Dimap(std::string const &nomFile)
    {
        lireDimapFile(nomFile);
    }
    
    void lireDimapFile(std::string const &nomFile);
    
    Pt2dr direct(Pt2dr Pimg, double altitude)const;
    
    Pt2dr indirect(Pt2dr Pgeo, double altitude)const;
    
    void createDirectGrid(double ulcSamp, double ulcLine,
                          double stepSamp, double stepLine,
                          int nbSamp, int  nbLine,
                          std::vector<double> const &vAltitude,
                          std::vector<Pt2dr> &vPtCarto)const;
    
    void createIndirectGrid(double ulcX, double ulcY,
                            double stepX, double stepY,
                            int nbX, int  nbY,
                            std::vector<double> const &vAltitude,
                            std::vector<Pt2dr> &vPtImg)const;
    
    
    
    void createGrid(std::string const &nomGrid,
                    double ulcSamp, double ulcLine,
                    double stepSamp, double stepLine,
                    int nbSamp, int  nbLine,
                    std::vector<double> vAltitude)const;
    void info()
    {
        std::cout << "long_scale   : "<<long_scale<<  " | long_off   : "<<long_off<<std::endl;
        std::cout << "lat_scale    : "<<lat_scale<<   " | lat_off    : "<<lat_off<<std::endl;
        std::cout << "height_scale : "<<height_scale<<" | height_off : "<<height_off<<std::endl;
        std::cout << "samp_scale   : "<<samp_scale<<  " | samp_off   : "<<samp_off<<std::endl;
        std::cout << "line_scale   : "<<line_scale<<  " | line_off   : "<<line_off<<std::endl;
        std::cout << "first_row    : "<<first_row<<   " | last_row   : "<<last_row<<std::endl;
        std::cout << "first_col    : "<<first_col<<   " | last_col   : "<<last_col<<std::endl;
        std::cout << "first_lon    : "<<first_lon<<   " | last_lon   : "<<last_lon<<std::endl;
        std::cout << "first_lat    : "<<first_lat<<   " | last_lat   : "<<last_lat<<std::endl;
    }
    
//private:
    std::vector<double> direct_samp_num_coef;
    std::vector<double> direct_samp_den_coef;
    std::vector<double> direct_line_num_coef;
    std::vector<double> direct_line_den_coef;
    
    std::vector<double> indirect_samp_num_coef;
    std::vector<double> indirect_samp_den_coef;
    std::vector<double> indirect_line_num_coef;
    std::vector<double> indirect_line_den_coef;
    
    std::vector<double> direct_bias;
    std::vector<double> indirect_bias;

    double first_row;
    double first_col;
    double last_row;
    double last_col;
    double first_lon;
    double first_lat;
    double last_lon;
    double last_lat;
    
    double long_scale;
    double long_off;
    double lat_scale;
    double lat_off;

    double samp_scale;
    double samp_off;
    double line_scale;
    double line_off;

    double height_scale;
    double height_off;
    
};

Pt2dr Dimap::direct(Pt2dr Pimg, double altitude)const
{
    //Calcul des coordonnées image normalisées
    double Y=(Pimg.y-line_off)/line_scale;
    double X=(Pimg.x-samp_off)/samp_scale;
    double Z=(altitude-height_off)/height_scale;
    double vecteurD[]={1,X,Y,Z,Y*X,X*Z,Y*Z,X*X,Y*Y,Z*Z,X*Y*Z,X*X*X,Y*Y*X,X*Z*Z,X*X*Y,Y*Y*Y,Y*Z*Z,X*X*Z,Y*Y*Z,Z*Z*Z};
    
    double long_den = 0.;
    double long_num = 0.;
    double lat_den = 0.;
    double lat_num = 0.;
    
    for (int i=0; i<20; i++)
    {
        lat_num  += vecteurD[i]*direct_line_num_coef[i];
        lat_den  += vecteurD[i]*direct_line_den_coef[i];
        long_num += vecteurD[i]*direct_samp_num_coef[i];
        long_den += vecteurD[i]*direct_samp_den_coef[i];
    }
    
    //Calcul final
    Pt2dr Pgeo;
    if ((lat_den != 0) &&
        (long_den !=0))
    {
        Pgeo.x = (lat_num / lat_den) * lat_scale + lat_off;
        Pgeo.y = (long_num / long_den) * long_scale + long_off;
    }
    else
    {
        std::cout << "Erreur de calcul - dénominateur nul";
    }
    return Pgeo;
}

Pt2dr Dimap::indirect(Pt2dr Pgeo, double altitude)const
{
    //Calcul des coordonnées image normalisées
    double Y=(Pgeo.y-long_off)/long_scale;
    double X=(Pgeo.x-lat_off)/lat_scale;
    double Z=(altitude-height_off)/height_scale;
    double vecteurD[]={1,Y,X,Z,Y*X,Y*Z,X*Z,Y*Y,X*X,Z*Z,X*Y*Z,Y*Y*Y,Y*X*X,Y*Z*Z,X*Y*Y,X*X*X,X*Z*Z,Y*Y*Z,X*X*Z,Z*Z*Z};
    
    double samp_den = 0.;
    double samp_num = 0.;
    double line_den = 0.;
    double line_num = 0.;
    
    for (int i=0; i<20; i++)
    {
        line_num  += vecteurD[i]*indirect_line_num_coef[i];
        line_den  += vecteurD[i]*indirect_line_den_coef[i];
        samp_num  += vecteurD[i]*indirect_samp_num_coef[i];
        samp_den  += vecteurD[i]*indirect_samp_den_coef[i];
    }
    //Calcul final
    Pt2dr Pimg;
    if ((samp_den != 0) &&
        (line_den !=0))
    {
        Pimg.x = (samp_num / samp_den) * samp_scale + samp_off;
        Pimg.y = (line_num / line_den) * line_scale + line_off;
    }
    else
    {
        std::cout << "Erreur de calcul - dénominateur nul";
    }
    return Pimg;
}

void Dimap::createDirectGrid(double ulcSamp, double ulcLine,
                      double stepSamp, double stepLine,
                      int nbSamp, int  nbLine,
                      std::vector<double> const &vAltitude,
                      std::vector<Pt2dr> &vPtCarto)const
{
    vPtCarto.clear();
    // On cree un fichier de point geographique pour les transformer avec proj4
    {
        std::ofstream fic("ptGeo.txt");
        fic << std::setprecision(9);
        for(size_t i=0;i<vAltitude.size();++i)
        {
            double altitude = vAltitude[i];
            for(int l=0;l<nbLine;++l)
            {
                for(int c = 0;c<nbSamp;++c)
                {
                    Pt2dr Pimg(ulcSamp + c * stepSamp, ulcLine + l * stepLine);
                    Pt2dr Pgeo = direct(Pimg,altitude);
                    fic << Pgeo.y <<" "<<Pgeo.x<<";"<<std::endl;
                }
            }
        }
    }
    // transfo en Lambert93
    system("cs2cs +proj=latlon +datum=WGS84 +ellps=WGS84 +to +init=IGNF:LAMB93 -s ptGeo.txt > ptLamb93.txt");
    // chargement des points
    std::ifstream fic("ptLamb93.txt");
    while(!fic.eof()&&fic.good())
    {
        double X,Y,Z;
        char c;
        fic >> Y >> X >> Z >> c;
        if (fic.good())
            vPtCarto.push_back(Pt2dr(X,Y));
    }
    std::cout << "Nombre de points lus : "<<vPtCarto.size()<<std::endl;
}

void Dimap::createIndirectGrid(double ulcX, double ulcY,
                        double stepX, double stepY,
                        int nbX, int  nbY,
                        std::vector<double> const &vAltitude,
                        std::vector<Pt2dr> &vPtImg)const
{
    vPtImg.clear();
    // On cree un fichier de point geographique pour les transformer avec proj4
    {
        std::ofstream fic("ptLamb93.txt");
        fic << std::setprecision(9);
        for(int l=0;l<nbY;++l)
        {
            double Y = ulcY + l*stepY;
            for(int c = 0;c<nbX;++c)
            {
                double X =ulcX + c*stepX;
                fic << Y <<" "<<X<<";"<<std::endl;
            }
        }
    }
    // transfo en Geo
    system("cs2cs +init=IGNF:LAMB93 +to +proj=latlon +datum=WGS84 +ellps=WGS84 -f %.12f -s ptLamb93.txt > ptGeo.txt");
    for(size_t i=0;i<vAltitude.size();++i)
    {
        double altitude = vAltitude[i];
        // chargement des points
        std::ifstream fic("ptGeo.txt");
        while(!fic.eof()&&fic.good())
        {
            double lon ,lat ,Z;
            char c;
            fic >> lon  >> lat >> Z >> c;
            if (fic.good())
            {
                vPtImg.push_back(indirect(Pt2dr(lat,lon),altitude));
            }
        }
    }
    std::cout << "Nombre de points lus : "<<vPtImg.size()<<std::endl;
}



void Dimap::createGrid(std::string const &nomGrid,
                double ulcSamp, double ulcLine,
                double stepSamp, double stepLine,
                int nbSamp, int  nbLine,
                std::vector<double> vAltitude)const
{
    /*
     double ulcX, double ulcY,
     double stepX, double stepY,
     int nbX, int  nbY,
     */
    QString nomImage("Image");
    
    std::vector<Pt2dr> vPtCarto;
    createDirectGrid(ulcSamp,ulcLine,stepSamp,stepLine,nbSamp,nbLine,vAltitude,vPtCarto);
    
    double xmin,xmax,ymin,ymax;
    xmin = vPtCarto[0].x;
    xmax = xmin;
    ymin = vPtCarto[0].y;
    ymax = ymin;
    for(size_t i=1;i<vPtCarto.size();++i)
    {
        Pt2dr const & pt = vPtCarto[i];
        if (xmin>pt.x) xmin = pt.x;
        else if (xmax<pt.x) xmax = pt.x;
        if (ymin>pt.y) ymin = pt.y;
        else if (ymax<pt.y) ymax = pt.y;
    }

    std::cout << "Emprise carto : "<<xmin<<" "<<ymin<<" "<<xmax<<" "<<ymax<<std::endl;
    double stepX = stepSamp*0.5;
    double stepY = stepX;
    std::vector<Pt2dr> vPtImg;
    createIndirectGrid(xmin,ymin,stepX,stepY,
                       (xmax-xmin)/stepX,(ymax-ymin)/stepY,vAltitude,vPtImg);
    
    ///Création de la grille
    QFile grille (nomGrid.c_str());
    //création du flux d'écriture
    QXmlStreamWriter ecrire (&grille);
    //Ouverture du fichier avec test de l'entrée
    if (!grille.open(QIODevice::WriteOnly | QIODevice::Text))
    { return;
        qDebug()<< "boucle erreur entrée sortie";
    }
    
    //Mise en forme
    ecrire.setAutoFormatting(true);
    
    //Choix du codec: x Mac Roan a été remplacé par UTF-8 sur les Macs
    //Écriture de l'en-tête : <?xml version="1.0" encoding="UTF-8" ?>
    ecrire.writeStartDocument();
    
    // Création de la racine
    ecrire.writeStartElement("trans_coord_grid");
    ecrire.writeAttribute("version","5");
    ecrire.writeAttribute("name","");
    
    //Récupération et écriture du groupe date heure
    time_t t= time(0);
    struct tm * timeInfo =localtime(&t);
    QString date;
    QChar fillchar='0';
    date=QString("%1%2").arg(timeInfo-> tm_year+1900).arg(timeInfo-> tm_mon,2,10,fillchar);
    date=date+QString("%1%2").arg(timeInfo-> tm_mday,2,10,fillchar).arg(timeInfo-> tm_hour,2,10,fillchar);
    date=date+QString("%1%2").arg(timeInfo-> tm_min,2,10,fillchar).arg(timeInfo-> tm_sec,2,10,fillchar);
    ecrire.writeTextElement("date",date);
    
    // Création d'un élément trans_coord
    
    ecrire.writeStartElement("trans_coord");
    ecrire.writeAttribute("name","");
    
    // Création d'un élément trans_sys_coord
    ecrire.writeStartElement("trans_sys_coord");
    ecrire.writeAttribute("name","");
    
    /// Création d'un élément sys_coord
    ecrire.writeStartElement("sys_coord");
    ecrire.writeAttribute("name","sys1");
    // Création d'un élément
    ecrire.writeStartElement("sys_coord_plani");
    ecrire.writeAttribute("name","sys1");
    ecrire.writeTextElement("code",nomImage);
    ecrire.writeTextElement("unit","p");
    ecrire.writeTextElement("direct","0");  ///0
    ecrire.writeTextElement("sub_code","*");    ///
    ecrire.writeTextElement("vertical",nomImage);  ///
    // Fermeture l'élément
    ecrire.writeEndElement();
    // Création d'un élément
    ecrire.writeStartElement("sys_coord_alti");
    ecrire.writeAttribute("name","sys1");
    ecrire.writeTextElement("code","LAMBERT93");
    ecrire.writeTextElement("unit","m");
    // Fermeture l'élément
    ecrire.writeEndElement();
    /// Fermeture de l'élément sys_coord
    ecrire.writeEndElement();
    
    /// Création d'un élément sys_coord
    ecrire.writeStartElement("sys_coord");
    ecrire.writeAttribute("name","sys2");
    // Création d'un élément
    ecrire.writeStartElement("sys_coord_plani");
    ecrire.writeAttribute("name","sys2");
    ecrire.writeTextElement("code","LAMBERT93");
    ecrire.writeTextElement("unit","m");
    ecrire.writeTextElement("direct","1");
    ecrire.writeTextElement("sub_code","*");     ///
    ecrire.writeTextElement("vertical","LAMBERT93");
    // Fermeture l'élément
    ecrire.writeEndElement();
    // Création d'un élément
    ecrire.writeStartElement("sys_coord_alti");
    ecrire.writeAttribute("name","sys2");
    ecrire.writeTextElement("code","LAMBERT93");
    ecrire.writeTextElement("unit","m");
    // Fermeture l'élément
    ecrire.writeEndElement();
    /// Fermeture de l'élément sys_coord
    ecrire.writeEndElement();
    
    // Fermeture l'élément trans_sys_coord
    ecrire.writeEndElement();
    
    {
        ecrire.writeTextElement("category","0");
        ecrire.writeTextElement("type_modele","1");
        ecrire.writeTextElement("direct_available","1");
        ecrire.writeTextElement("inverse_available","0");
    }
    
    // Fermeture de l'élément trans_coord
    ecrire.writeEndElement();
    
    // Création d'un élément multi_grid
    ecrire.writeStartElement("multi_grid");
    ecrire.writeAttribute("version","1");
    ecrire.writeAttribute("name","1-2");
    
    QString sUlcSamp;
    sUlcSamp.setNum(ulcSamp, 'E',12);
    QString sUlcLine;
    sUlcLine.setNum(ulcLine, 'E',12);
    
    ecrire.writeTextElement("upper_left",sUlcSamp+"  "+sUlcLine);
    ecrire.writeTextElement("columns_interval",QString("%1").arg(stepSamp));
    ecrire.writeTextElement("rows_interval",QString("%1").arg(-stepLine));
    ecrire.writeTextElement("columns_number",QString("%1").arg(nbSamp));
    ecrire.writeTextElement("rows_number",QString("%1").arg(nbLine));
    ecrire.writeTextElement("components_number","2");
    
    std::vector<Pt2dr>::const_iterator it = vPtCarto.begin();
    
    for(size_t i=0;i<vAltitude.size();++i)
    {
        ecrire.writeStartElement("layer");
        ecrire.writeAttribute("value",QString("%1").arg(vAltitude[i]));
        
        for(int l=0;l<nbLine;++l)
        {
            for(int c = 0;c<nbSamp;++c)
            {
                Pt2dr const &PtCarto = (*it);
                ++it;
                QString sX;
                sX.setNum(PtCarto.x , 'E',12);
                QString sY;
                sY.setNum(PtCarto.y , 'E',12);
                
                ecrire.writeCharacters(sX+" "+sY+"\n");
            }
        }
        /// Fermeture de l'élément layer value
        ecrire.writeEndElement();
    }
    
    // Fermeture de l'élément de l'élément multi_grid
    ecrire.writeEndElement();
    
    // Fermeture de l'élément de l'élément trans_coord
    ecrire.writeEndElement();
    
    // Finalise le document grille
    ecrire.writeEndDocument();
    
    // Fermeture du fichier
    grille.close();
}

//Lecture du fichier DIMAP
void Dimap::lireDimapFile(std::string const &nomFile)
{
    direct_samp_num_coef.clear();
    direct_samp_den_coef.clear();
    direct_line_num_coef.clear();
    direct_line_den_coef.clear();
    
    indirect_samp_num_coef.clear();
    indirect_samp_den_coef.clear();
    indirect_line_num_coef.clear();
    indirect_line_den_coef.clear();
    
    //qDebug()<< "entrée dans la fonction lecture DIMAP";
    QFile dimapFile (nomFile.c_str());
    
    //Création de booléens
    bool Direct_Model;
    bool Direct_Model_Val;
    
    //Test de l'entrée dans le fichier
    if (!dimapFile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        qDebug()<< "echec ouverture du fichier Dimap";
        return;
    }
    
    //Définition du flux de lecture xml
    QXmlStreamReader lectureXML;
    lectureXML.setDevice(&dimapFile);
    
    //boucle de lecture du fichier xml
    while (!lectureXML.isEndDocument())
    {
        //qDebug()<<" début de la lecture du fichier";
        lectureXML.readNext();
        
        //test si on est un mot clé d'une famille d'information
        if (lectureXML.isStartElement())
        {
            //conversion du mot lu en QString
            QString motLu = lectureXML.name().toString();
            
            //test de lecture sur la nature du modèle considéré
            if ( (motLu.startsWith("Direct_Model",Qt::CaseInsensitive) == true))
            {
                //lecture de données sur le modèle direct
                Direct_Model=true;
            }
            if ( (motLu.startsWith("Inverse_Model",Qt::CaseInsensitive) == true))
            {
                //lecture de données sur le modèle indirect
                Direct_Model=false;
            }
            
            //Valeur du booléen sur les domaines de validité
            if ( motLu.endsWith("Domain", Qt::CaseInsensitive) == true)
            {
                //lecture de données sur le domaine de validité
                Direct_Model_Val=true;
            }
            if ( (motLu.endsWith("Domain", Qt::CaseInsensitive) == true) &&
                (motLu.startsWith("/", Qt::CaseInsensitive) == true))
            {
                //lecture de données sur le domaine de validité
                Direct_Model_Val=false;
            }
            
            //test sur le type de coefficient lu en fonction du début du nom
            if (Direct_Model == true)
            {
                if (motLu.startsWith("SAMP_NUM_COEFF", Qt::CaseInsensitive)) direct_samp_num_coef.push_back(lectureXML.readElementText().toDouble());
                if (motLu.startsWith("SAMP_DEN_COEFF", Qt::CaseInsensitive)) direct_samp_den_coef.push_back(lectureXML.readElementText().toDouble());
                if (motLu.startsWith("LINE_NUM_COEFF", Qt::CaseInsensitive)) direct_line_num_coef.push_back(lectureXML.readElementText().toDouble());
                if (motLu.startsWith("LINE_DEN_COEFF", Qt::CaseInsensitive)) direct_line_den_coef.push_back(lectureXML.readElementText().toDouble());
            }
            else
            {
                if (motLu.startsWith("SAMP_NUM_COEFF", Qt::CaseInsensitive)) indirect_samp_num_coef.push_back(lectureXML.readElementText().toDouble());
                if (motLu.startsWith("SAMP_DEN_COEFF", Qt::CaseInsensitive)) indirect_samp_den_coef.push_back(lectureXML.readElementText().toDouble());
                if (motLu.startsWith("LINE_NUM_COEFF", Qt::CaseInsensitive)) indirect_line_num_coef.push_back(lectureXML.readElementText().toDouble());
                if (motLu.startsWith("LINE_DEN_COEFF", Qt::CaseInsensitive)) indirect_line_den_coef.push_back(lectureXML.readElementText().toDouble());
            }
            
            if (motLu.startsWith("FIRST_ROW", Qt::CaseInsensitive)) first_row = lectureXML.readElementText().toDouble();
            if (motLu.startsWith("FIRST_COL", Qt::CaseInsensitive)) first_col = lectureXML.readElementText().toDouble();
            if (motLu.startsWith("LAST_ROW", Qt::CaseInsensitive)) last_row = lectureXML.readElementText().toDouble();
            if (motLu.startsWith("LAST_COL", Qt::CaseInsensitive)) last_col = lectureXML.readElementText().toDouble();
            if (motLu.startsWith("FIRST_LON", Qt::CaseInsensitive)) first_lon = lectureXML.readElementText().toDouble();
            if (motLu.startsWith("FIRST_LAT", Qt::CaseInsensitive)) first_lat = lectureXML.readElementText().toDouble();
            if (motLu.startsWith("LAST_LON", Qt::CaseInsensitive)) last_lon = lectureXML.readElementText().toDouble();
            if (motLu.startsWith("LAST_LAT", Qt::CaseInsensitive)) last_lat = lectureXML.readElementText().toDouble();
            
            //Récupération des scales et offset dans l'ordre long, lat, height, samp, line
            if (motLu.startsWith ("LONG_SCALE", Qt::CaseInsensitive)) long_scale = lectureXML.readElementText().toDouble();
            if (motLu.startsWith ("LONG_OFF", Qt::CaseInsensitive)) long_off = lectureXML.readElementText().toDouble();
            if (motLu.startsWith ("LAT_SCALE", Qt::CaseInsensitive)) lat_scale = lectureXML.readElementText().toDouble();
            if (motLu.startsWith ("LAT_OFF", Qt::CaseInsensitive)) lat_off = lectureXML.readElementText().toDouble();
            if (motLu.startsWith ("HEIGHT_SCALE", Qt::CaseInsensitive)) height_scale = lectureXML.readElementText().toDouble();
            if (motLu.startsWith ("HEIGHT_OFF", Qt::CaseInsensitive)) height_off = lectureXML.readElementText().toDouble();
            if (motLu.startsWith ("SAMP_SCALE", Qt::CaseInsensitive)) samp_scale = lectureXML.readElementText().toDouble();
            if (motLu.startsWith ("SAMP_OFF", Qt::CaseInsensitive)) samp_off = lectureXML.readElementText().toDouble();
            if (motLu.startsWith ("LINE_SCALE", Qt::CaseInsensitive)) line_scale = lectureXML.readElementText().toDouble();
            if (motLu.startsWith ("LINE_OFF", Qt::CaseInsensitive)) line_off = lectureXML.readElementText().toDouble();
        }
        //test si on est à la fin d'une famille d'informations
        else if (lectureXML.isEndElement())
        {
            lectureXML.readNext();
        }
        
        //En cas d'erreur
        if (lectureXML.hasError())
        {
            qDebug() << "erreur de lecture";
            
        }
    }
    
    dimapFile.close();
}



int Dimap2Grid_main(int argc, char **argv) {
    std::string aNameFileDimap;// fichier Dimap
    std::string aNameFileGrid;// fichier GRID
    double altiMin;
    double altiMax;
    int nbLayers;
    
    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aNameFileDimap,"Dimap file")
                   << EAMC(aNameFileGrid,"Grid file")
     << EAMC(altiMin,"altitude min")
     << EAMC(altiMax,"altitude max")
     << EAMC(nbLayers,"number of layers")
     ,
        LArgMain()
     );
    
    Dimap dimap(aNameFileDimap);
    std::cout << "Dimap info:"<<std::endl;
    std::cout << "=============================="<<std::endl;
    dimap.info();
    std::cout << "=============================="<<std::endl;
    
    std::vector<double> vAltitude;
    for(int i=0;i<nbLayers;++i)
        vAltitude.push_back(altiMin+(altiMax-altiMin)/(nbLayers-1));
    dimap.createGrid(aNameFileGrid,
                     dimap.first_col,dimap.first_row,128,128,(dimap.last_col-dimap.first_col)/128,(dimap.last_row-dimap.first_row)/128,
                     vAltitude);
    
    return EXIT_SUCCESS;
}
#endif

