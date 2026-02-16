import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial.distance import cdist

from lossfuncs import EntropyLoss, RegressionLoss

from .modules import Discriminator, FeatBottleneck, FeatClassifier, ReverseLayerF


class DiversifyReg(nn.Module):
    def __init__(
        self,
        featurizer,
        feature_dim,
        latent_domain_num,
        bottleneck_dim=256,
        dis_hidden=256,
        classifier_type="linear",
        layer="bn",
        alpha=0.1,
        alpha1=0.1,
        lam=0.0,
        reg_loss="mse",
        y_mean=None,
        y_std=None,
    ):
        super().__init__()
        self.featurizer = featurizer
        self.alpha = alpha
        self.alpha1 = alpha1
        self.lam = lam
        self.reg_loss = reg_loss
        self.reg_loss_fn = RegressionLoss(reg_loss)
        self.latent_domain_num = latent_domain_num
        self.n_bins = 4
        boundaries = None
        if y_mean is not None and y_std is not None:
            y_mean = float(y_mean)
            y_std = float(y_std)
            if not np.isfinite(y_std) or y_std <= 0:
                y_std = 1.0
            boundaries = torch.tensor(
                [y_mean - y_std, y_mean, y_mean + y_std], dtype=torch.float32
            )
        self.register_buffer("bin_boundaries", boundaries)

        self.dbottleneck = FeatBottleneck(feature_dim, bottleneck_dim, layer_type=layer)
        self.ddiscriminator = Discriminator(bottleneck_dim, dis_hidden, 1)
        self.dclassifier = FeatClassifier(latent_domain_num, bottleneck_dim, layer_type=classifier_type)

        self.bottleneck = FeatBottleneck(feature_dim, bottleneck_dim, layer_type=layer)
        self.discriminator = Discriminator(bottleneck_dim, dis_hidden, latent_domain_num)
        self.regressor = nn.Linear(bottleneck_dim, 1)

        self.abottleneck = FeatBottleneck(feature_dim, bottleneck_dim, layer_type=layer)
        self.aregressor = nn.Linear(bottleneck_dim, latent_domain_num * self.n_bins)

    def _prepare_batch(self, batch, device):
        x, y, dlabels, _, pdlabels, mask, _ = batch
        x = x.to(device)
        y = y.to(device).float()
        dlabels = dlabels.to(device).long()
        pdlabels = pdlabels.to(device).long()
        mask = mask.to(device).bool() if mask is not None else None
        return x, y, dlabels, pdlabels, mask

    @staticmethod
    def _flatten(feats, y, dlabels, pdlabels, mask):
        if feats.dim() == 2:
            feats_flat = feats
            y_flat = y
            d_flat = dlabels
            pd_flat = pdlabels
            mask_flat = mask
        else:
            feats_flat = feats.reshape(-1, feats.shape[-1])
            y_flat = y.reshape(-1)
            d_flat = dlabels.reshape(-1)
            pd_flat = pdlabels.reshape(-1)
            mask_flat = mask.reshape(-1) if mask is not None else None

        if mask_flat is not None:
            mask_flat = mask_flat.bool()
            if mask_flat.sum().item() == 0:
                return feats_flat[:0], y_flat[:0], d_flat[:0], pd_flat[:0]
            feats_flat = feats_flat[mask_flat]
            y_flat = y_flat[mask_flat]
            d_flat = d_flat[mask_flat]
            pd_flat = pd_flat[mask_flat]

        return feats_flat, y_flat, d_flat, pd_flat

    def update_a(self, batch, optimizer, device):
        self.train()
        x, y, dlabels, pdlabels, mask = self._prepare_batch(batch, device)
        feats = self.featurizer(x)
        feats_flat, y_flat, _, pd_flat = self._flatten(feats, y, dlabels, pdlabels, mask)
        if feats_flat.numel() == 0:
            return {"class": 0.0}
        if self.bin_boundaries is None:
            raise ValueError("bin_boundaries is not set; provide y_mean and y_std")

        y_bin = torch.bucketize(y_flat, self.bin_boundaries)
        combined = pd_flat * self.n_bins + y_bin

        z = self.abottleneck(feats_flat)
        preds = self.aregressor(z)
        loss = F.cross_entropy(preds, combined)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"class": float(loss.item())}

    def update_d(self, batch, optimizer, device):
        self.train()
        x, y, dlabels, pdlabels, mask = self._prepare_batch(batch, device)
        feats = self.featurizer(x)
        feats_flat, y_flat, _, pd_flat = self._flatten(feats, y, dlabels, pdlabels, mask)
        if feats_flat.numel() == 0:
            return {"total": 0.0, "dis": 0.0, "ent": 0.0}

        z = self.dbottleneck(feats_flat)
        disc_in = ReverseLayerF.apply(z, self.alpha1)
        disc_out = self.ddiscriminator(disc_in)
        if disc_out.shape != y_flat.shape:
            y_target = y_flat.view_as(disc_out)
        else:
            y_target = y_flat
        # disc_loss = F.mse_loss(disc_out, y_target)
        disc_loss = self.reg_loss_fn(disc_out, y_target)

        cd_out = self.dclassifier(z)
        ent_loss = EntropyLoss.entropy_logits(cd_out) * self.lam + F.cross_entropy(cd_out, pd_flat)
        loss = ent_loss + disc_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"total": float(loss.item()), "dis": float(disc_loss.item()), "ent": float(ent_loss.item())}

    def update(self, batch, optimizer, device):
        self.train()
        x, y, dlabels, pdlabels, mask = self._prepare_batch(batch, device)
        feats = self.featurizer(x)

        feats_flat, y_flat, _, pd_flat = self._flatten(feats, y, dlabels, pdlabels, mask)
        if feats_flat.numel() == 0:
            return {"total": 0.0, "reg": 0.0, "dis": 0.0}

        z = self.bottleneck(feats_flat)
        disc_in = ReverseLayerF.apply(z, self.alpha)
        disc_out = self.discriminator(disc_in)
        disc_loss = F.cross_entropy(disc_out, pd_flat)

        pred = self.regressor(z).squeeze(-1)
        reg_loss = self.reg_loss_fn(pred, y_flat)
        loss = reg_loss + disc_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"total": float(loss.item()), "reg": float(reg_loss.item()), "dis": float(disc_loss.item())}

    def set_dlabel(self, loader, device):
        self.dbottleneck.eval()
        self.dclassifier.eval()
        self.featurizer.eval()

        all_fea = []
        all_output = []
        all_index = []

        with torch.no_grad():
            for batch in loader:
                x, y, dlabels, pdlabels, mask = self._prepare_batch(batch, device)
                feats = self.featurizer(x)
                feats_flat, _, _, _ = self._flatten(feats, y, dlabels, pdlabels, mask)

                if feats_flat.numel() == 0:
                    continue

                z = self.dbottleneck(feats_flat)
                outputs = self.dclassifier(z)
                all_fea.append(z.cpu())
                all_output.append(outputs.cpu())

                index = batch[-1]
                mask_cpu = batch[-2] if len(batch) > 1 else None
                if mask_cpu is not None:
                    index = index.reshape(-1)[mask_cpu.reshape(-1)].cpu()
                else:
                    index = index.reshape(-1).cpu()
                all_index.append(index.numpy())

        if not all_fea:
            self.dbottleneck.train()
            self.dclassifier.train()
            self.featurizer.train()
            return

        all_fea = torch.cat(all_fea, dim=0).float().cpu().numpy()
        all_output = torch.cat(all_output, dim=0)
        all_output = torch.softmax(all_output, dim=1).float().cpu().numpy()
        all_index = np.concatenate(all_index)

        all_fea = np.concatenate([all_fea, np.ones((all_fea.shape[0], 1), dtype=all_fea.dtype)], axis=1)
        all_fea = all_fea / (np.linalg.norm(all_fea, axis=1, keepdims=True) + 1e-12)

        k = all_output.shape[1]
        aff = all_output
        initc = aff.T.dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dist = cdist(all_fea, initc, metric="cosine")
        pred_label = dist.argmin(axis=1)

        for _ in range(1):
            aff = np.eye(k)[pred_label]
            initc = aff.T.dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dist = cdist(all_fea, initc, metric="cosine")
            pred_label = dist.argmin(axis=1)

        dataset = loader.dataset
        if not hasattr(dataset, "set_labels_by_index"):
            raise AttributeError("Dataset does not support set_labels_by_index")
        dataset.set_labels_by_index(pred_label, all_index, label_type="pdlabel")

        self.dbottleneck.train()
        self.dclassifier.train()
        self.featurizer.train()

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            feats = self.featurizer(x)
            if feats.dim() == 2:
                z = self.bottleneck(feats)
                return self.regressor(z).squeeze(-1)

            flat = feats.reshape(-1, feats.shape[-1])
            z = self.bottleneck(flat)
            pred = self.regressor(z).squeeze(-1)
            return pred.view(feats.shape[0], feats.shape[1])
