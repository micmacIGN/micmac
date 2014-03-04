#ifndef TREE_H
#define TREE_H

#include <QStandardItemModel>

#include "StdAfx.h"

using namespace NS_SaisiePts;

class TreeItem
{
public:
    explicit TreeItem(const QList<QVariant> &data, TreeItem *parent = 0);
    ~TreeItem();

    void appendChild(TreeItem *child);

    TreeItem *child(int row);
    int childCount() const;
    int columnCount() const;
    QVariant data(int column) const;
    int row() const;
    TreeItem *parent();

private:
    QList<TreeItem*> childItems;
    QList<QVariant> itemData;
    TreeItem *parentItem;
};

class TreeModel : public QAbstractItemModel
{
    Q_OBJECT

public:
    explicit TreeModel(QObject *parent = 0);
    ~TreeModel();

    QVariant        data(const QModelIndex &index, int role) const;
    bool            setData(const QModelIndex &index, const QVariant &value, int role = Qt::EditRole);
    Qt::ItemFlags   flags(const QModelIndex &index) const;
    QVariant        headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const;
    QModelIndex     index(int row, int column, const QModelIndex &parent = QModelIndex()) const;
    QModelIndex     parent(const QModelIndex &index) const;

    int     rowCount    (const QModelIndex &parent = QModelIndex()) const;
    int     columnCount (const QModelIndex &parent = QModelIndex()) const;

    void    setAppli(cAppli_SaisiePts* appli);

signals:
    void dataChanged();

private:
    void    setupModelData(TreeItem *parent);

    TreeItem *rootItem;

    cAppli_SaisiePts* _appli;
};

#endif // TREE_H
