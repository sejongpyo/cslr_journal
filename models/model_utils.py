import torch

def make_adj_mat(key_siged, device):
    with torch.no_grad():
        rounded = torch.round(torch.sigmoid(key_siged.squeeze(-1))).cpu()
        adj_mat = torch.zeros(rounded.size(0), rounded.size(1), rounded.size(1))
        for i in range(len(rounded)):
            mat = (rounded[i].unsqueeze(0) * torch.t(rounded[i].unsqueeze(0)))
            mat -= torch.eye(mat.size(0))
            diag_mat = torch.ones(mat.size(0)-1)
            diag_mat = torch.diag(diag_mat, 1)
            diag_mat += torch.t(diag_mat)
            mat += diag_mat
            mat[mat==-1], mat[mat==2] = 0, 1
            mat += torch.eye(rounded.size(1)) # add self loop
            adj_mat[i] = mat
            
    return adj_mat.to(device)

def make_norm_adj_mat(key_siged, device):
    with torch.no_grad():
        rounded = torch.round(torch.sigmoid(key_siged.squeeze(-1))).cpu()
        adj_mat = torch.zeros(rounded.size(0), rounded.size(1), rounded.size(1))
        for i in range(len(rounded)):
            mat = (rounded[i].unsqueeze(0) * torch.t(rounded[i].unsqueeze(0)))
            mat -= torch.eye(mat.size(0))
            diag_mat = torch.ones(mat.size(0)-1)
            diag_mat = torch.diag(diag_mat, 1)
            diag_mat += torch.t(diag_mat.clone())
            mat += diag_mat
            mat[mat==-1], mat[mat==2] = 0, 1
            mat += torch.eye(rounded.size(1)) # add self loop
            # D**(-0.5) @ ADJ @ D**(-0.5)
            degree = torch.sum(mat, dim=1)
            degree = torch.pow(degree, -0.5)
            degree_mat = torch.diag(degree)
            norm_mat = torch.matmul(torch.matmul(degree_mat, mat), degree_mat)
            
            adj_mat[i] = norm_mat
            
    return adj_mat.to(device)

def make_sel_mat(key_siged, device):
    with torch.no_grad():
        rounded = torch.round(torch.sigmoid(key_siged.squeeze(-1))).cpu()
        adj_mat = torch.zeros(rounded.size(0), rounded.size(1), rounded.size(1))
        for i in range(len(rounded)):
            mat = (rounded[i].unsqueeze(0) * torch.t(rounded[i].unsqueeze(0)))
            mat -= torch.eye(mat.size(0))
            diag_mat = torch.ones(mat.size(0)-1)
            diag_mat = torch.diag(diag_mat, 1)
            diag_mat += torch.t(diag_mat.clone())
            mat += diag_mat
            mat[mat==-1], mat[mat==2] = 0, 1
            mat += torch.diag(rounded[i])
            adj_mat[i] = mat
            
    return adj_mat.to(device)

def make_sel_mat_2(key_siged, prob_sizes, device):
    with torch.no_grad():
        x = torch.round(torch.sigmoid(key_siged.squeeze(-1))).cpu()
        adj_mat = torch.zeros(x.size(0), x.size(1), x.size(1))
        diag_mat = torch.ones(x.size(1)-1)
        diag_mat = torch.diag(diag_mat, 1)
        diag_mat += torch.t(diag_mat.clone())
        for i in range(len(x)):
            combined = diag_mat + torch.diag(x[i])
            adj_mat[i] = combined        
        for j in range(len(prob_sizes)):
            adj_mat[j, prob_sizes[j]:, :] = 0
            adj_mat[j, :, prob_sizes[j]:] = 0
                    
    return adj_mat.to(device)

def make_sel_mat_3(key_siged, prob_sizes, device): # 앞뒤 키포인트 일경우에만 + 키포인트만 셀프
    with torch.no_grad():
        x = torch.round(torch.sigmoid(key_siged.squeeze(-1))).cpu()
        adj_mat = torch.zeros(x.size(0), x.size(1), x.size(1))
        
        for i in range(len(x)):
            adj_mat[i] = torch.diag(x[i])
            
        for j in range(len(prob_sizes)):
            diag_mat = torch.zeros(x.size(1)-1)
            for k in range(prob_sizes[j]-1):
                if x[j][k] == 1 and x[j][k+1] == 1:
                    diag_mat[k] = 1
            diag_mat = torch.diag(diag_mat, 1)
            diag_mat += torch.t(diag_mat.clone())
                
            adj_mat[j] += diag_mat
                                
            adj_mat[j, prob_sizes[j]:, :] = 0
            adj_mat[j, :, prob_sizes[j]:] = 0
        
        adj_mat = adj_mat.to(device)
            
    return adj_mat

def simple_adj_mat(x, device):
    with torch.no_grad():
        adj_mat = torch.zeros(x.size(0), x.size(1), x.size(1))
        diag_mat = torch.ones(x.size(1)-1)
        diag_mat = torch.diag(diag_mat, 1)
        diag_mat += torch.t(diag_mat.clone())
        diag_mat += torch.eye(x.size(1))
        adj_mat += diag_mat
        adj_mat = adj_mat.to(device)
    
    return adj_mat

if __name__=="__main__":
    import torch.nn as nn
    # out = torch.randn(2, 6, 1)
    # m = nn.Sigmoid()
    # output = m(out)
    # make_adj_mat(output)
    
    out = torch.randn(2, 6, 512)
    print(simple_adj_mat(out, "cpu").shape)
    print(simple_adj_mat(out, "cpu").dtype)