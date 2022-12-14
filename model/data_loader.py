import torch
from torch.utils.data import Dataset, IterableDataset

class meshiRefinementDataset(Dataset):
    """Dataset of refinements from AlphaFold and RosettaFold."""

    def __init__(self, root_dir, crop_size=None, split='train', type='AlphaFold', homothresh=0.9, debug = False,deb_examp = 200):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.homothresh = homothresh    # homologic threshold
        self.type = type                # Dataset type of no native(AlphaFold\RosettaFold)
        self.split = split              # split type - if not train no split
        self.root_dir = root_dir        # root dir
        self.cropSize = crop_size       # crop size
        self.proteinPath = self.root_dir     # proteion root path
        self.debug = debug              # debuge mode
        self.deb_exmp = deb_examp       # debug examples
        self.load_data()
        

        self.typeindices = range(len(self.ids))
        if type is not None:
            self.typeindices = []
            for ii, id in enumerate(self.ids):
                if (self.type is not None) and (self.type in id):
                    self.typeindices.append(ii)

        
    def load_data(self):
        if self.debug:
            self.seq = torch.load(self.proteinPath + '/seq.pt')[:self.deb_examp]
            self.ids = torch.load(self.proteinPath + '/ids.pt') [:self.deb_examp]
            self.coordN = torch.load(self.proteinPath + '/CoordN.pt')[:self.deb_examp]
            self.coordAlpha = torch.load(self.proteinPath + '/CoordAlpha.pt') [:self.deb_examp]
            self.coordC = torch.load(self.proteinPath + '/CoordC.pt') [:self.deb_examp]
            self.coordBeta = torch.load(self.proteinPath + '/CoordBeta.pt')  [:self.deb_examp]

            self.nativemask = torch.load(self.proteinPath + '/nativemask.pt') [:self.deb_examp]
            self.msk = torch.load(self.proteinPath + '/mask.pt') [:self.deb_examp]

            self.gdtts = torch.load(self.proteinPath + '/GDTTS.pt') [:self.deb_examp]
            self.iddts = torch.load(self.proteinPath + '/IDDTS.pt')[:self.deb_examp]

            self.coordNNative = torch.load(self.proteinPath + '/CoordNNative.pt') [:self.deb_examp]
            self.coordAlphaNative = torch.load(self.proteinPath + '/CoordCaNative.pt') [:self.deb_examp]
            self.coordCNative = torch.load(self.proteinPath + '/CoordCNative.pt') [:self.deb_examp]
            self.coordBetaNative = torch.load(self.proteinPath + '/CoordCbNative.pt')[:self.deb_examp]

            self.embeddings = torch.load(self.proteinPath + '/embeddings.pt') [:self.deb_examp]
        else:
            self.seq = torch.load(self.proteinPath + '/seq.pt') 
            self.ids = torch.load(self.proteinPath + '/ids.pt')
            self.coordN = torch.load(self.proteinPath + '/CoordN.pt')  # [:200]
            self.coordAlpha = torch.load(self.proteinPath + '/CoordAlpha.pt')  # [:200]
            self.coordC = torch.load(self.proteinPath + '/CoordC.pt')  # [:200]
            self.coordBeta = torch.load(self.proteinPath + '/CoordBeta.pt')  # [:200]

            self.nativemask = torch.load(self.proteinPath + '/nativemask.pt')  # [:200]
            self.msk = torch.load(self.proteinPath + '/mask.pt')  # [:200]

            self.gdtts = torch.load(self.proteinPath + '/GDTTS.pt')  # [:200]
            self.iddts = torch.load(self.proteinPath + '/IDDTS.pt')  # [:200]

            self.coordNNative = torch.load(self.proteinPath + '/CoordNNative.pt')  # [:200]
            self.coordAlphaNative = torch.load(self.proteinPath + '/CoordCaNative.pt')  # [:200]
            self.coordCNative = torch.load(self.proteinPath + '/CoordCNative.pt')  # [:200]
            self.coordBetaNative = torch.load(self.proteinPath + '/CoordCbNative.pt')  # [:200]

            self.embeddings = torch.load(self.proteinPath + '/embeddings.pt')  # [:200] 

    def __len__(self):
        if self.type is None:
            return len(self.seq)
        else:
            return len(self.typeindices)

    def toTestSplit(self):
        self.split = 'test'

    def toTrainSplit(self):
        self.split = 'train'

    def read_protein_data(self, i):
        # self.proteinPath is the path to the folder with all the .pt files of the proteins
        self.proteinPath = self.root_dir
        i = self.typeindices[i]
        seq = self.seq[i]
        ids = self.ids[i]

        coordN = self.coordN[i]
        coordAlpha = self.coordAlpha[i]
        # if coordAlpha.max()>5e4:
        #    print('problems in Calpha')
        #    print('in data loader 1')
        coordC = self.coordC[i]
        coordBeta = self.coordBeta[i]

        nativemask = self.nativemask[i]
        msk = self.msk[i]

        gdtts = self.gdtts[i]
        iddts = self.iddts[i]

        coordNNative = self.coordNNative[i]
        coordAlphaNative = self.coordAlphaNative[i]
        coordCNative = self.coordCNative[i]
        coordBetaNative = self.coordBetaNative[i]

        embeddings = self.embeddings[i]

        return coordN, coordAlpha, coordC, coordBeta, seq, ids, msk, gdtts, iddts, embeddings, coordNNative, coordAlphaNative, coordCNative, coordBetaNative, nativemask

    def __getitem__(self, idx):
        ok = False
        while not ok:
            coordN, coordAlpha, coordC, coordBeta, seq, id, msk, gdtt, \
            iddt, embedding, coordNNative, coordAlphaNative, \
            coordCNative, coordBetaNative, nativemask = self.read_protein_data(idx)

            dt = torch.get_default_dtype()
            coordN = coordN.to(dt)
            coordAlpha = coordAlpha.to(dt)
            coordC = coordC.to(dt)
            coordBeta = coordBeta.to(dt)
            seq = seq.to(dt)
            embedding = embedding.to(dt)
            coordNNative = coordNNative.to(dt)
            coordAlphaNative = coordAlphaNative.to(dt)
            coordCNative = coordCNative.to(dt)
            coordBetaNative = coordBetaNative.to(dt)

            s = seq.mean(-1)
            if (self.homothresh is not None) and (s.max() > self.homothresh):
                # print("protein is too homogenuous", id)
                idx = (idx + 1) % self.__len__()
                continue

            if (self.type is not None) and (not self.type in id):
                # print("id not from ", self.type, " ,", id)
                # idx = torch.randint(self.__len__(), (1, 1))[0]
                idx = (idx + 1) % self.__len__()
                continue

            scale = 1e-2
            Mnat = nativemask
            M = msk & Mnat

            ind = torch.where(M)[0]
            istart = ind[0]
            ilast = ind[-1]
            M = M[istart:ilast + 1]
            msk = msk[istart:ilast + 1]
            msk = msk.type('torch.FloatTensor')
            if torch.any(msk == 0):
                # print("id problem", id)
                idx = (idx + 1) % self.__len__()
                continue
            else:
                ok = True

        Mnat = Mnat[istart:ilast + 1]
        Mnat = Mnat.type('torch.FloatTensor')

        X1 = coordAlpha.t()
        X2 = coordC.t()
        X3 = coordN.t()
        X4 = coordBeta.t()

        glyIndices = torch.where(X4[0, :] > 5e4)[0]
        X4[:, glyIndices] = getCB(X3[:, glyIndices], X1[:, glyIndices], X2[:, glyIndices])

        X1native = coordAlphaNative.t()
        X2native = coordCNative.t()
        X3native = coordNNative.t()
        X4native = coordBetaNative.t()

        X4native[:, glyIndices] = getCB(X3native[:, glyIndices], X1native[:, glyIndices], X2native[:, glyIndices])

        CoordsNative = scale * torch.stack((X1native, X2native, X3native, X4native), dim=1)
        CoordsNative = CoordsNative.type('torch.FloatTensor')
        CoordsNative = CoordsNative[:, :, istart:ilast + 1]

        Coords = scale * torch.stack((X1, X2, X3, X4), dim=1)
        Coords = Coords.type('torch.FloatTensor')
        Coords = Coords[:, :, istart:ilast + 1]
        A0 = seq.t()
        A = A0[istart:ilast + 1, :]

        embedding0 = embedding.clone()
        embedding0 = embedding0[istart:ilast + 1, :]

        Coords = Coords.unsqueeze(0)
        CoordsNative = CoordsNative.unsqueeze(0)

        # Coords = Coords[:, :, :, M == 1]
        # CoordsNative = CoordsNative[:, :, :, M == 1]
        # A = A[M == 1, :]
        # embedding0 = embedding0[M == 1, :]
        if (self.cropSize is not None) and (self.split == 'train'):
            nnodes = Coords.shape[-1]
            CoordsBatch = torch.zeros(1, 3, 4, 2 * self.cropSize, device=Coords.device)
            CoordsDecoy = torch.zeros(1, 3, 4, 2 * self.cropSize, device=Coords.device)
            ABatch = torch.zeros(1, 2 * self.cropSize, 20, device=Coords.device)
            if nnodes > (2 * self.cropSize + 2):
                mid = nnodes // 2
                istart1 = torch.randint(0, mid - self.cropSize, (1,))
                istart2 = torch.randint(mid, nnodes - self.cropSize, (1,))
                Patch1 = torch.arange(istart1[0], (istart1[0] + self.cropSize))
                Patch2 = torch.arange(istart2[0], (istart2[0] + self.cropSize))
            else:
                istart1 = torch.randint(0, nnodes, (1,))
                istart2 = torch.randint(0, nnodes, (1,))
                Patch1 = torch.arange(istart1[0], (istart1[0] + self.cropSize))
                Patch2 = torch.arange(istart2[0], (istart2[0] + self.cropSize))
                # Cyclical:
                Patch1 = Patch1 % nnodes
                Patch2 = Patch2 % nnodes

            # Take patch:
            p = torch.cat((Patch1, Patch2), dim=0)
            CoordsBatch[0, :, :, :] = CoordsNative[0, :, :, p]
            CoordsDecoy[0, :, :, :] = Coords[0, :, :, p]
            ABatch[0, :, :] = A[p, :]
            embeddingBatch = embedding0[p, :].unsqueeze(0)

            Mnat = Mnat[p]
            # M = torch.ones(2 * self.cropSize, device=Coords.device)

        else:
            CoordsDecoy = Coords
            CoordsBatch = CoordsNative
            ABatch = A.unsqueeze(0)
            embeddingBatch = embedding0.unsqueeze(0)

        nodalFeat = ABatch
        nodalFeat = nodalFeat.transpose(1, 2)
        nodalFeat = nodalFeat.to(torch.float32)

        z1, z2 = lengthConstraints(CoordsBatch)
        MTet = (z1.abs() < 1.0)
        MTetTet = (z2.abs() < 1.0)

        g = MTet.sum(dim=[2, 3])
        g = (g == 16).unsqueeze(2).unsqueeze(3)
        MTet = g * MTet

        g = MTetTet.sum(dim=[3])
        g = (g == 4).unsqueeze(3)
        MTetTet = g * MTetTet

        CoordsDecoy = CoordsDecoy.transpose(1, 2)
        CoordsBatch = CoordsBatch.transpose(1, 2)

        return id, nodalFeat, CoordsBatch, Mnat.unsqueeze(
            0), ABatch, CoordsDecoy, MTet, MTetTet, gdtt, iddt, embeddingBatch
