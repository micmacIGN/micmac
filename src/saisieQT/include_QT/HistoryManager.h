#ifndef _HISTORYMANAGER_H
#define _HISTORYMANAGER_H

#include "Elise_QT.h"
#include <iostream>

struct selectInfos
{
    selectInfos(){}
    selectInfos(QVector <QPointF> pol,int mode)
    {
        poly = pol;
        selection_mode = mode;
    }
    //! polyline infos
    QVector <QPointF> poly;

    //! selection mode
    int         selection_mode;

    GLdouble    mvmatrix[16];
    GLdouble    projmatrix[16];
    GLint       glViewport[4];
};

class HistoryManager
{
public:

    HistoryManager();

    void    push_back(selectInfos &infos);

    //! Get the selection infos stack
    QVector <selectInfos> getSelectInfos(){ return _infos; }

	selectInfos& getSelectInfo(int id){ return _infos[id]; }

    int    getActionIdx(){ return _actionIdx; }

    int    size() { return _infos.size(); }

    void   undo() { if (_actionIdx > 0)  _actionIdx--; }

    void   redo() { if (_actionIdx < _infos.size()) _actionIdx++; }

    void   reset(){ _actionIdx = 0; _initSize = 0; _infos.clear(); }

    bool   load(QString filename);  //return true si Fichier existe && contient un bon xml
    void   save();

    bool   sizeChanged(){ return _infos.size() != _initSize; }

private:
    //! selection infos stack
    QVector <selectInfos> _infos;

    //! current action index
    int        _actionIdx;

    //! initial number of actions loaded
    int        _initSize;

    QString    _filename;
};

#endif
