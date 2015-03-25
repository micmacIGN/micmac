///
///
///
template        <class T>
BufferImage<T>::BufferImage(int NRows, int NLines, int NBands):_size(NRows,NLines),_numBands(NBands)
{
    _data = new T[NRows*NLines*NBands];
    _pixelSpace = NBands;
    _lineSpace = NRows*NBands;
    _bandSpace = 1;
    _delete=true;
}

template        <class T>
BufferImage<T>::BufferImage(int NRows, int NLines, int NBands,T* data,int pixelSpace,int lineSpace, int bandSpace):_data(data),_size(NRows,NLines),_numBands(NBands),_pixelSpace(pixelSpace),_lineSpace(lineSpace),_bandSpace(bandSpace),_delete(false)
{
}

template        <class T>
BufferImage<T>::BufferImage():_size(0,0),_numBands(0)
{
    _data = NULL;
    _pixelSpace = 0;
    _lineSpace = 0;
    _bandSpace = 1;
    _delete=false;
}


///
///
///
template        <class T>
BufferImage<T>::~BufferImage()
{
    bool verbose = false;
    if(verbose) std::cout<<"[BufferImage<T>::~BufferImage] "<<std::endl;
    if ((_delete) && (_data)){
        
        if(verbose) std::cout<<"[BufferImage<T>::~BufferImage] calling delete[] _data "<<std::endl;
        
        delete[] _data;
    }
    
}

///
///
///
template        <class T>
T* BufferImage<T>::getPtr()
{
    return _data;
}

///
///
///
template        <class T>
T const * BufferImage<T>::getPtr()const
{
    return _data;
}

///
///
///
template        <class T>
int BufferImage<T>::getPixelSpace()const
{
    return _pixelSpace;
}

///
///
///
template        <class T>
int BufferImage<T>::getLineSpace()const
{
    return _lineSpace;
}

///
///
///
template        <class T>
int BufferImage<T>::getBandSpace()const
{
    return _bandSpace;
}

#ifndef IGN_WITH_DEBUG
#define IGN_WITH_DEBUG 0
#endif
///
///
///
template        <class T>
T* BufferImage<T>::getLinePtr(int L)
{
#if IGN_WITH_DEBUG
    if ((L<(int)0)||(L>(int)(_size.second)))
    {
        std::ostringstream oss;
        oss << __FILE__ << " : "<<__LINE__;
        throw std::invalid_argument(oss.str());
    }
#endif
    return _data+L*_lineSpace;
}

///
///
///
template        <class T>
T const * BufferImage<T>::getLinePtr(int L)const
{
#if IGN_WITH_DEBUG
    if ((L<0)||(L>(int)_size.second))
    {
        std::ostringstream oss;
        oss << __FILE__ << " : "<<__LINE__;
        throw std::invalid_argument(oss.str());
    }
#endif
    return _data+L*_lineSpace;
}

///
///
///
template        <class T>
T BufferImage<T>::operator () (int C,int L, int Band)const
{
#if IGN_WITH_DEBUG
    if ((C<0)||(C>(int)_size.first) || (L<0)||(L>(int)_size.second) || (Band<0)||(Band>(int)_numBands))
    {
        std::cout<<"C L Band == "<<C<<" "<<L<<" "<<Band<<std::endl;
        std::cout<<"_size == "<<_size.first<<" "<<_size.second<<std::endl;
        std::ostringstream oss;
        oss << __FILE__ << " : "<<__LINE__;
        std::cout << oss.str();
        throw std::invalid_argument(oss.str());
    }
#endif
    return *(_data+L*_lineSpace+C*_pixelSpace+Band*_bandSpace);
}

///
///
///
template        <class T>
T & BufferImage<T>::operator ()(int C,int L, int Band)
{
#if IGN_WITH_DEBUG
    if ((C<0)||(C>(int)_size.first) || (L<0)||(L>(int)_size.second) || (Band<0)||(Band>(int)_numBands))
    {
        std::cout<<"C L Band == "<<C<<" "<<L<<" "<<Band<<std::endl;
        std::cout<<"_size == "<<_size.first<<" "<<_size.second<<std::endl;
        std::ostringstream oss;
        oss << __FILE__ << " : "<<__LINE__;
        std::cout << oss.str()<<std::endl;
        throw std::invalid_argument(oss.str());
    }
#endif
    return *(_data+L*_lineSpace+C*_pixelSpace+Band*_bandSpace);
}


///
///
///
template        <class T>
std::pair<size_t,size_t> BufferImage<T>::size(int deZoom)const
{
    return _size;
}

///
///
///
template        <class T>
size_t						BufferImage<T>::numCols()const
{
    return _size.first;
}

///
///
///
template        <class T>
size_t						BufferImage<T>::numLines()const
{
    return _size.second;
}

///
///
///
template        <class T>
size_t BufferImage<T>::numBands()const
{
    return _numBands;
}

///
///
///
template        <class T>
bool BufferImage<T>::operator == (BufferImage<T> const &other)const
{
    if (this->size()!=other.size())
        return false;
    if (this->numBands()!=other.numBands())
        return false;
    int NC,NL;
    // this->Size(NC,NL);
    std::pair<size_t,size_t> sz = this->size();
    NC=sz.first; NL=sz.second;
    int NbBands = this->numBands();
    for(int l=0;l<NL;++l)
    {
        for(int c=0;c<NC;++c)
        {
            for(int k=0;k<NbBands;++k)
            {
                if ((*this)(c,l,k)!=other(c,l,k))
                    return false;
            }
        }
    }
    return true;
}

///
///
///
template        <class T>
BufferImage<T>& BufferImage<T>::operator /= (double val)
{
    int NC = _size.first;
    int NL = _size.second;
    T* ptrLine = _data;
    for(int l=0;l<NL;++l)
    {
        T* ptrPixel = ptrLine;
        for(int c=0;c<NC;++c)
        {
            T* ptrBand = ptrPixel;
            for(int k=0;k<(int)_numBands;++k)
            {
                (*ptrBand)=(T)((*ptrBand)/val);
                ptrBand+=_bandSpace;
            }
            ptrPixel+=_pixelSpace;
        }
        ptrLine+=_lineSpace;
    }
    return *this;
}

///
///
///
template        <class T>
BufferImage<T>& BufferImage<T>::operator *= (double val)
{
    int NC = _size.first;
    int NL = _size.second;
    T* ptrLine = _data;
    for(int l=0;l<NL;++l)
    {
        T* ptrPixel = ptrLine;
        for(int c=0;c<NC;++c)
        {
            T* ptrBand = ptrPixel;
            for(int k=0;k<(int)_numBands;++k)
            {
                (*ptrBand)=(T)((*ptrBand)*val);
                ptrBand+=_bandSpace;
            }
            ptrPixel+=_pixelSpace;
        }
        ptrLine+=_lineSpace;
    }
    return *this;
}

///
///
///
template        <class T>
BufferImage<T>& BufferImage<T>::operator += (T val)
{
    int NC = _size.first;
    int NL = _size.second;
    T* ptrLine = _data;
    for(int l=0;l<NL;++l)
    {
        T* ptrPixel = ptrLine;
        for(int c=0;c<NC;++c)
        {
            T* ptrBand = ptrPixel;
            for(int k=0;k<(int)_numBands;++k)
            {
                (*ptrBand)=(T)((*ptrBand)+val);
                ptrBand+=_bandSpace;
            }
            ptrPixel+=_pixelSpace;
        }
        ptrLine+=_lineSpace;
    }
    return *this;
}

///
///
///
template        <class T>
BufferImage<T>& BufferImage<T>::operator -= (T val)
{
    int NC = _size.first;
    int NL = _size.second;
    T* ptrLine = _data;
    for(int l=0;l<NL;++l)
    {
        T* ptrPixel = ptrLine;
        for(int c=0;c<NC;++c)
        {
            T* ptrBand = ptrPixel;
            for(size_t k=0;k<_numBands;++k)
            {
                (*ptrBand)=(T)((*ptrBand)-val);
                ptrBand+=_bandSpace;
            }
            ptrPixel+=_pixelSpace;
        }
        ptrLine+=_lineSpace;
    }
    return *this;
}

///
///
///
template        <class T>
BufferImage<T>& BufferImage<T>::operator += (BufferImage<T> const &img)
{
    int NC = _size.first;
    int NL = _size.second;
    int NC2,NL2;
    //img.Size(NC2,NL2);
    NC2 = img.numCols();
    NL2 = img.numLines();
    int nbBands2 = img.numBands();
    T* ptrLine = _data;
    const T* ptrLine2 = img.getPtr();
    for(int l=0;(l<NL)&&(l<NL2);++l)
    {
        T* ptrPixel = ptrLine;
        const T* ptrPixel2 = ptrLine2;
        for(int c=0;c<NC;++c)
        {
            if (c<NC2)
            {
                T* ptrBand = ptrPixel;
                const T* ptrBand2 = ptrPixel2;
                for(size_t k=0;k<_numBands;++k)
                {
                    if ((int)k<nbBands2)
                    {
                        (*ptrBand)=(T)((*ptrBand)+(*ptrBand2));
                        ptrBand2+=img.getBandSpace();
                    }
                    ptrBand+=_bandSpace;
                }
                ptrPixel2+=img.getPixelSpace();
            }
            ptrPixel+=_pixelSpace;
        }
        ptrLine+=_lineSpace;
        ptrLine2+=img.getLineSpace();
    }
    return *this;
}

///
///
///
template        <class T>
BufferImage<T>& BufferImage<T>::operator -= (BufferImage<T> const &img)
{
    int NC = _size.first;
    int NL = _size.second;
    int NC2,NL2;
    img.Size(NC2,NL2);
    int nbBands2 = img.numBands();
    T* ptrLine = _data;
    const T* ptrLine2 = img.getPtr();
    for(int l=0;(l<NL)&&(l<NL2);++l)
    {
        T* ptrPixel = ptrLine;
        const T* ptrPixel2 = ptrLine2;
        for(int c=0;c<NC;++c)
        {
            if (c<NC2)
            {
                T* ptrBand = ptrPixel;
                const T* ptrBand2 = ptrPixel2;
                for(int k=0;k<_numBands;++k)
                {
                    if (k<nbBands2)
                    {
                        (*ptrBand)=(T)((*ptrBand)-(*ptrBand2));
                        ptrBand2+=img.getBandSpace();
                    }
                    ptrBand+=_bandSpace;
                }
                ptrPixel2+=img.getPixelSpace();
            }
            ptrPixel+=_pixelSpace;
        }
        ptrLine+=_lineSpace;
    }
    return *this;
}

///
///
///
template        <class T>
BufferImage<T>& BufferImage<T>::operator *= (BufferImage<T> const &img)
{
    int NC = _size.first;
    int NL = _size.second;
    int NC2,NL2;
    img.Size(NC2,NL2);
    int nbBands2 = img.numBands();
    T* ptrLine = _data;
    const T* ptrLine2 = img.getPtr();
    for(int l=0;(l<NL)&&(l<NL2);++l)
    {
        T* ptrPixel = ptrLine;
        const T* ptrPixel2 = ptrLine2;
        for(int c=0;c<NC;++c)
        {
            if (c<NC2)
            {
                T* ptrBand = ptrPixel;
                const T* ptrBand2 = ptrPixel2;
                for(int k=0;k<_numBands;++k)
                {
                    if (k<nbBands2)
                    {
                        (*ptrBand)=(T)((*ptrBand)*(*ptrBand2));
                        ptrBand2+=img.getBandSpace();
                    }
                    ptrBand+=_bandSpace;
                }
                ptrPixel2+=img.getPixelSpace();
            }
            ptrPixel+=_pixelSpace;
        }
        ptrLine+=_lineSpace;
        ptrLine2+=img.getLineSpace();
    }
    return *this;
}

///
///
///
template        <class T>
BufferImage<T>& BufferImage<T>::operator /= (BufferImage<T> const &img)
{
    int NC = _size.first;
    int NL = _size.second;
    int NC2,NL2;
    //img.Size(NC2,NL2);
    NC2 = img.numCols();
    NL2 = img.numLines();
    int nbBands2 = img.numBands();
    T* ptrLine = _data;
    const T* ptrLine2 = img.getPtr();
    for(int l=0;(l<NL)&&(l<NL2);++l)
    {
        T* ptrPixel = ptrLine;
        const T* ptrPixel2 = ptrLine2;
        for(int c=0;c<NC;++c)
        {
            if (c<NC2)
            {
                T* ptrBand = ptrPixel;
                const T* ptrBand2 = ptrPixel2;
                for(int k=0;k<(int)_numBands;++k)
                {
                    if (k<nbBands2)
                    {
                        if ((*ptrBand2)!=0)
                            (*ptrBand)=(T)((*ptrBand)/(*ptrBand2));
                        ptrBand2+=img.getBandSpace();
                    }
                    ptrBand+=_bandSpace;
                }
                ptrPixel2+=img.getPixelSpace();
            }
            ptrPixel+=_pixelSpace;
        }
        ptrLine+=_lineSpace;
        ptrLine2+=img.getLineSpace();
    }
    return *this;
}

///
///
///
template        <class T>
void BufferImage<T>::initialize(int NRows,int NLines, int NBands)
{
    bool verbose = 0;
    if(verbose)	std::cout << "BufferImage<T>::initialize" << std::endl;
    _size=std::pair<size_t,size_t>(NRows,NLines);
    _numBands=NBands;
    if(verbose)	std::cout << _size.first << "x" << _size.second << "x" << _numBands << std::endl;
    if ((_data)&&(_delete))
    {
        if (verbose) std::cout<<"delete old data at adress: "<<_data<<std::endl;
        delete[] _data;
    }
    if (NRows*NLines>0)
    {
        if(verbose)	std::cout << "Allocation du buffer" << std::endl;
        _data = new T[NRows*NLines*NBands];
        if (_data==NULL)
        {
            std::cout << "BufferImage : Erreur d'allocation d'un buffer de "<<NRows<<" x "<<NLines<<" x "<<NBands<<std::endl;
        }
        if(verbose)	std::cout << "Buffer correctement alloue" << std::endl;
    }
    else
        _data = NULL;
    
    _pixelSpace = NBands;
    _lineSpace = NRows*NBands;
    _bandSpace = 1;
    _delete=true;
}


///
///
///
template        <class T>
void BufferImage<T>::initialize(int NRows,int NLines, int NBands, const T &defaultValue)
{
    this->initialize(NRows,NLines,NBands);
    
    //initialisation des valeurs :
    *(this) = (T)defaultValue;
}

///
///
///
template        <class T>
BufferImage<T>& BufferImage<T>::operator = (T val)
{
    int NC = _size.first;
    int NL = _size.second;
    T* ptrLine = _data;
    for(int l=0;l<NL;++l)
    {
        T* ptrPixel = ptrLine;
        for(int c=0;c<NC;++c)
        {
            T* ptrBand = ptrPixel;
            for(int k=0;k<(int)_numBands;++k)
            {
                (*ptrBand)=(T)(val);
                ptrBand+=_bandSpace;
            }
            ptrPixel+=_pixelSpace;
        }
        ptrLine+=_lineSpace;
    }
    return *this;
}


///
///
///
template        <class T>
BufferImage<T>& BufferImage<T>::operator = (BufferImage<T> const &ori)
{
    //std::cout<<"Attention, recopie d'image \n" ;
    std::pair<size_t,size_t> taille = ori.size(1);
    int nbBands = ori.numBands();
    this->initialize(taille.first,taille.second,nbBands);
    int NC = _size.first;
    int NL = _size.second;
    T* ptrLine = _data;
    for(int l=0;l<NL;++l)
    {
        T* ptrPixel = ptrLine;
        for(int c=0;c<NC;++c)
        {
            T* ptrBand = ptrPixel;
            for(unsigned int k=0;k<_numBands;++k)
            {
                (*ptrBand)=(T)ori(c,l,k);
                ptrBand+=_bandSpace;
            }
            ptrPixel+=_pixelSpace;
        }
        ptrLine+=_lineSpace;
    }
    return *this;
}
