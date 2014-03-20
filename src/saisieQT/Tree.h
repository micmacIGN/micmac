#ifndef TREE_H
#define TREE_H

#include <QAbstractTableModel>
#include <QColor>

#include "StdAfx.h"

using namespace NS_SaisiePts;


class ModelPointGlobal : public QAbstractTableModel
{
    Q_OBJECT
public:

    ModelPointGlobal(QObject *parent, cAppli_SaisiePts* appli);

    int             rowCount(const QModelIndex &parent = QModelIndex()) const ;

    int             columnCount(const QModelIndex &parent = QModelIndex()) const;

    QVariant        data(const QModelIndex &index, int role = Qt::DisplayRole) const;

    QVariant        headerData(int section, Qt::Orientation orientation, int role) const;

    bool            setData(const QModelIndex & index, const QVariant & value, int role = Qt::EditRole);

    Qt::ItemFlags   flags(const QModelIndex &index) const;

    bool            insertRows(int row, int count, const QModelIndex & parent = QModelIndex());

    bool            caseIsSaisie(int idRow);

signals:

    void            pGChanged();

protected:

    int             CountPG_CaseName() const;

    int             CountPG() const;

    int             CountCaseNamePoint() const;

private:

    cAppli_SaisiePts* mAppli;

};

class ModelCImage : public QAbstractTableModel
{
    Q_OBJECT
public:

    ModelCImage(QObject *parent, cAppli_SaisiePts* appli);

    int             rowCount(const QModelIndex &parent = QModelIndex()) const ;

    int             columnCount(const QModelIndex &parent = QModelIndex()) const;

    QVariant        data(const QModelIndex &index, int role = Qt::DisplayRole) const;

    QVariant        headerData(int section, Qt::Orientation orientation, int role) const;

    bool            setData(const QModelIndex & index, const QVariant & value, int role = Qt::EditRole);

    Qt::ItemFlags   flags(const QModelIndex &index) const;

    bool            insertRows(int row, int count, const QModelIndex & parent = QModelIndex());

    int             getIdGlobSelect() const;

    void            setIdGlobSelect(int value);

private:

    cAppli_SaisiePts* mAppli;

    int             idGlobSelect;

};

#endif // TREE_H
