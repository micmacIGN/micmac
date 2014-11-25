#ifndef __BUFFERIMAGE_H__
#define __BUFFERIMAGE_H__

#include <complex>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <map>
#include <memory>


///\brief Classe proposee pour gerer les images en memoire
template < class T >
class BufferImage {
public:
    
    ///\brief Constructeur par defaut: buffer de taille nulle*/
    BufferImage();
    
    ///\brief Destructeur avec liberation de la memoire*/
    virtual				~BufferImage();
    
    ///\brief Constructeur a  partir d'un buffer memoire deja  existant
    /// attention le buffer n'est pas recopie: on utilise directement le pointeur data
    ///comme le pointeur a ete fourni en entree il n'est pas sous la responsabilite de cette classe
    ///et il ne sera pas desalloue dans le destructeur
    BufferImage(int nXSize, int nYSize, int nBands);
    
    BufferImage(int nXSize,int nYSize, int nNBands,
                T* data, int nPixelSpace, int nLineSpace, int nBandSpace);
    
    
    /////////////METHODES HERITEES DE IMAGE/////////////
    ///\brief taille du buffer
    std::pair<size_t,size_t>						size(int aDeZoom=1)const;
    ///\brief nombre de colonnes
    size_t							numCols()const;
    ///\brief nombre de lignes
    size_t							numLines()const;
    ///\brief Nombre de bandes
    size_t							numBands()const;

    /////////////METHODES PROPRES A LA CLASSE/////////////
    ///\brief Modifier la taille du buffer: attention les donnees sont reinitialisees
    void							initialize(int NRows, int NLines, int NBands, const T& defaultValue);
    ///\brief Modifier la taille du buffer: attention les donnees ne sont pas reinitialisee
    void							initialize(int NRows, int NLines, int NBands);
    
    ///ACCES A LA STRUCTURE MEMOIRE
    ///\brief Pointeur vers le debut du bloc memoire
    T*								getPtr();
    ///\brief Pointeur vers le debut du bloc memoire
    T const *						getPtr()const;
    ///\brief decalage memoire pour passer d'un pixel au suivant (typiquement NBands)*/
    int								getPixelSpace()const;
    ///\brief decalage memoire pour passer d'une ligne a  la suivante (typiquement NRows*NBands)*/
    int								getLineSpace()const;
    ///\brief decalage memoire pour passer d'un canal au suivant (typiquement 1)*/
    int								getBandSpace()const;
    ///\brief pointeur vers le debut de la ligne L
    T*								getLinePtr(int L);
    ///\brief pointeur vers le debut de la ligne L
    T const *						getLinePtr(int L)const;
    
    
    
    ///OPERATORS
    ///\brief acces a  un pixel*/
    T								operator () (int C,int L, int Band=0)const;
    
    ///\brief acces a  un pixel
    T &								operator ()(int C,int L, int Band=0);
    
    ///\brief Operateur de comparaison (taille et pixel a pixel)
    bool							operator ==(BufferImage<T> const &other)const;
    
    ///\brief Operation sur l'ensemble des pixels
    BufferImage<T>&					operator /= (double val);
    
    ///\brief Operation sur l'ensemble des pixels
    BufferImage<T>&					operator *= (double val);
    
    ///\brief Operation sur l'ensemble des pixels
    BufferImage<T>&					operator += (T val);
    
    ///\brief Operation sur l'ensemble des pixels
    BufferImage<T>&					operator -= (T val);
    
    ///\brief Operation sur l'ensemble des pixels
    BufferImage<T>&					operator += (BufferImage<T> const &img);
    
    ///\brief Operation sur l'ensemble des pixels
    BufferImage<T>&					operator -= (BufferImage<T> const &img);
    
    ///\brief Operation sur l'ensemble des pixels
    BufferImage<T>&					operator *= (BufferImage<T> const &img);
    
    ///\brief Operation sur l'ensemble des pixels
    BufferImage<T>&					operator /= (BufferImage<T> const &img);
    
    ///\brief Operation sur l'ensemble des pixels
    BufferImage<T>&					operator = (T val);
    
    ///\brief Recopie d'une image
    BufferImage<T>&					operator = (BufferImage<T> const &ori);
    
protected:
    ///\brief pointeurs vers les pixels
    T*							_data;
    ///\brief taille de l'image en memoire
    std::pair<size_t,size_t>					_size;
    ///\brief nombre de bandes de l'image
    size_t						_numBands;
    ///\brief entrelacement des pixels
    int							_pixelSpace;
    ///\brief entrelacement des lignes
    int							_lineSpace;
    ///\brief entrelacement des bandes
    int							_bandSpace;
    ///\brief
    bool						_delete;
};

#include "BufferImage_code.h"

#endif
