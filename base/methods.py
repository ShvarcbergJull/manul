import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn

from sklearn.metrics import roc_curve

def prebording_data(data):
    avg = []
    var = []
    i = 0

    while i < len(data[0]):
        variance = np.var(data[:, i])
        if variance == 0:
            data = np.delete(data, i, 1)
            continue
        var.append(variance)
        avg.append(np.average(data[:, i]))
        i += 1
    
    return data, avg, var

def test_function(graph, f_x):
    part1 = np.dot(f_x, graph.my_laplassian)
    result = np.dot(part1, f_x.T)

    return result

def take_nn(train_features, train_target, dims, num_epochs, batch_size, model_settings=None, add_loss_func=None, graph=None, val=1):
    def baseline(dim):
        baseline_model = nn.Sequential(
            nn.Linear(dim, 512, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(512, 256, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(256, 256, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(256, 64, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(64, 1, dtype=torch.float64),
            nn.Sigmoid()
            # nn.LogSoftmax(dim=1)
        )

        return baseline_model
    

    if model_settings:
        b_model = model_settings["model"]
        criterion = model_settings["criterion"]
        optimizer = model_settings["optimizer"]
    else:
        b_model = baseline(dims)
        criterion = nn.BCELoss()
        # criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(b_model.parameters(), lr=1e-4, eps=1e-4)
    
    b_model.train()
    min_loss, t = np.inf, 0
    threshold = None
    lmd = 1/((batch_size - 100) ** 2)
    epoch = 0
    end = False
    last_loss = None
    count_loss = 0
    # adding_loss = None
    # if add_loss_func:
    #     adding_loss = test_function(graph, train_features)
    while epoch < num_epochs and end == False:
        permutation = torch.randperm(train_features.size()[0])
        loss_list = []
        for i in range(0, len(train_target), batch_size):
            indices = permutation[i:i+batch_size]
            # print(indices)
            batch_x, target_y = train_features[indices], train_target[indices]
            target_y = target_y.to(torch.float64)
            optimizer.zero_grad()
            output = b_model(batch_x)
            # output[output>0.5] = 1
            # output[output<=0.5] = 0
            # print(output.shape)
            loss = criterion(output, target_y.reshape_as(output))
            if add_loss_func:
                add_loss = add_loss_func(graph, output.detach().numpy(), indices)
                # add_loss = adding_loss[indices]
                try:
                    loss += lmd * torch.tensor(add_loss[0, 0])
                except:
                    loss += lmd * torch.tensor(add_loss)
                # loss += lmd * torch.tensor(add_loss)
            fpr, tpr, thresholds = roc_curve(target_y.reshape(-1), output.detach().numpy().reshape(-1))
            gmeans = np.sqrt(tpr * (1-fpr))
            ix = np.argmax(gmeans)
            # print("IX", thresholds[ix])
            if not threshold:
                threshold = thresholds[ix]
            else:
                threshold = np.mean([thresholds[ix], threshold])
            # loss = torch.mean(torch.abs(target_y-output))
            # loss = np.mean(np.abs(output - (target_y.reshape_as(output)).detach().numpy()))
            
            # print(loss)
            loss.backward()
            optimizer.step()
            # print(loss.item())
            loss_list.append(loss.item())
        # print(loss_list)
        loss_mean = np.mean(loss_list)

        if graph:
            if t == 0:
                last_loss = loss
            else:
                if np.isclose(loss.detach().numpy(), last_loss.detach().numpy(), atol=1e-3):
                    count_loss += 1
                    print("test")
                last_loss = loss
            if count_loss >= 10:
                end = True
        else:
            epoch += 1

        t += 1
        print('Surface training t={}, loss={}'.format(t, loss_mean), count_loss)

    b_model.eval()

    return {"model": b_model, "criterion": criterion, "optimizer": optimizer}, threshold, t

class Draw:

    def __init__(self, graph) -> None:
        self.graph = graph

    def draw_lowd(self, nodes):
        edges=[]
        for edge in self.graph.edges:
            edges.append(edge.prev.new_params)
            edges.append(edge.next.new_params)
            edges.append([None for i in range(len(edge.prev.new_params))])
        
        edges = np.array(edges).T
        edge_trace = go.Scatter(x=edges[0], y=edges[1], line=dict(width=4, color='#888'), hoverinfo='none', mode='lines')
        
        nodes = np.array([node.new_params for node in self.graph.nodes]).T
        colors = np.array([node.color for node in self.graph.nodes])
        node_trace = go.Scatter(x=nodes[0], y=nodes[1], mode='markers', hoverinfo='text',
                                  marker=dict(
                                      showscale=True,
                                      # colorscale options
                                      # #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                                      # #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                                      # #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                                      colorscale='YlGnBu',
                                      reversescale=True,
                                      color=colors,
                                      size=10,
                                  colorbar=dict(
                                    thickness=15,
                                    title='Node Connections',
                                    xanchor='left',
                                    titleside='right'
                                  ),
                                  line_width=2))
        
        return edge_trace, node_trace
    
    def draw_highd(self):

        nodes = []
        edges = []
        colors = []
        pairs = []

        for node in self.graph.nodes():
            nodes.append(node.params())
            colors.append(node.color())
            for neigh in node.neighbours():
                pairs.append((node.name(), neigh[1].name()))
                edges.append(node.params())
                edges.append(neigh[1].params())
                edges.append([None for i in range(len(node.params()))])         

        # nodes = np.array([node.params() for node in self.graph.nodes()]).T
        # colors = np.array([node.color() for node in self.graph.nodes()])

        # edges=[]
        # for edge in self.graph.edges:
        #     edges.append(self.graph.nodes[edge[0]]["params"])
        #     edges.append(self.graph.nodes[edge[1]]["params"])
        #     edges.append([None for i in range(len(self.graph.nodes[edge[0]]["params"]))])
        print(len(pairs))
        nodes = np.array(nodes).T
        edges = np.array(edges).T
    
        edge_trace = go.Scatter3d(x=edges[0], y=edges[1], z=edges[2], line=dict(width=4, color='#888'), hoverinfo='none', mode='lines')
        node_trace = go.Scatter3d(x=nodes[0], y=nodes[1], z=nodes[2], mode='markers', hoverinfo='text',
                                  marker=dict(
                                      showscale=True,
                                      # colorscale options
                                      # #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                                      # #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                                      # #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                                      colorscale='YlGnBu',
                                      reversescale=True,
                                      color=colors,
                                      size=10,
                                  colorbar=dict(
                                    thickness=15,
                                    title='Node Connections',
                                    xanchor='left',
                                    titleside='right'
                                  ),
                                  line_width=2))
        
        return edge_trace, node_trace

    def draw_graph(self, mode=0, data=None):

        if mode:
            edge_trace, node_trace = self.draw_lowd(data)
        else:
            edge_trace, node_trace = self.draw_highd()       
        
        fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
        fig.show()