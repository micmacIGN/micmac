#ifndef TREE_H
#define TREE_H

#include "QT_interface_Elise.h"

class cQT_Interface ;

class PointGlobalSFModel : public QSortFilterProxyModel
{
    Q_OBJECT

public:
    PointGlobalSFModel(QObject *parent = 0): QSortFilterProxyModel(parent){}

protected:

    bool filterAcceptsRow(int sourceRow, const QModelIndex &sourceParent) const;

    cAppli_SaisiePts* mAppli() const;
};

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

    cAppli_SaisiePts *getMAppli() const;

signals:

    void            pGChanged();

protected:

    int             AllCount() const;

    int             PG_Count() const;

    int             CaseNamePointCount() const;

private:

    cAppli_SaisiePts* mAppli;

    cQT_Interface*  _interface;

};


class ImagesSFModel : public QSortFilterProxyModel
{
    Q_OBJECT

public:
    ImagesSFModel(QObject *parent = 0): QSortFilterProxyModel(parent){}

protected:
    bool filterAcceptsRow(int sourceRow, const QModelIndex &sourceParent) const;

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

    cQT_Interface*  _interface;

};



#endif // TREE_H
