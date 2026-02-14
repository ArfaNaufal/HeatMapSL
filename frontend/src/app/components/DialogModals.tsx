import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from '@/app/components/ui/alert-dialog';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger, DialogClose } from '@/app/components/ui/dialog';
import { Button } from '@/app/components/ui/button';
import { Input } from "@/app/components/ui/input";
import { Label } from "@/app/components/ui/label";
import { useState } from 'react';
import { AlertTitle } from '@mui/material';
import { AlertDescription } from './ui/alert';
import { toast } from 'sonner';
import { API_URL } from '@/app/App';
import { Lock, Mail } from 'lucide-react';

interface ConfirmActionProps {
    trigger: React.ReactNode;
    title?: string;
    description?: string;
    actionText?: string;
    cancelText?: string;
    onConfirm: ()=>void;
    onCancel: ()=>void;
    variant?: "default" | "destructive";
};

interface UserData {
    email: string;
    password: string;
}

interface UserRegistrationDialogProps {
    trigger: React.ReactNode;
    handleSuccess: (data: UserData)=>void;
};

export const ConfirmCancelDialog = ({
    trigger,
    title = "Are you sure?",
    description = "This action can't be undone.",
    actionText = "Ok",
    cancelText = "Cancel",
    onConfirm,
    onCancel,
    variant = "default"
}: ConfirmActionProps) => {
    return (
        <AlertDialog>
            <AlertDialogTrigger asChild>{trigger}</AlertDialogTrigger>
            <AlertDialogContent className='bg-slate-900 border-white/10 text-white'>
                <AlertDialogHeader>
                    <AlertTitle>{title}</AlertTitle>
                    <AlertDescription className='text-white-500'>
                        {description}
                    </AlertDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                    <AlertDialogAction
                        onClick={onConfirm}
                        className={variant === "destructive" ? "bg-red-650 hover:bg-red-600" : "bg-cyan-600 hover:bg-cyan-700"}
                    >
                        {actionText}
                    </AlertDialogAction>
                    <AlertDialogCancel
                        onClick={onCancel}
                        className="bg-transparent border-white/10 text-white hover:bg-white/5"
                    >
                        {cancelText}
                    </AlertDialogCancel>
                </AlertDialogFooter>
            </AlertDialogContent>
        </AlertDialog>
    )
}

export const UserRegistrationDialog = ({
    trigger,
    handleSuccess = (data: UserData) => {}    
}: UserRegistrationDialogProps) => {
    const [isOpen, setIsOpen] = useState(false);
    const [email, setEmail] = useState<string>("");
    const [password, setPassword] = useState<string>("");

    
    const onFormSubmit = (e: React.FormEvent) => {
        e.preventDefault(); 
        handleSubmit();
    };

    const handleSubmit = async () => {
        const payload = { "email": email, "password": password };

        toast.promise(
            fetch(`${API_URL}/register`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('access_token')}`
                },
                body: JSON.stringify(payload)
            }).then(async (response) => {
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.message || 'Failed to register new user account.');
                }
                return response.json();
            }),
            {
                loading: 'Registering new user account...',
                success: (data) => {
                    if (window.location.pathname === '/login' && data.access_token && data.token_type == "bearer")
                        localStorage.setItem('access_token', data.access_token);
                    handleSuccess(payload);
                    setEmail("");
                    setPassword("");
                    setIsOpen(false); 
                    return 'User account registered!!';
                },
                error: 'Failed to register new user account.'
            }
        );
    }

    return (
        <Dialog open={isOpen} onOpenChange={setIsOpen}>
            <DialogTrigger asChild>
                {trigger}
            </DialogTrigger>
            <DialogContent className="sm:max-w-md bg-slate-900 border-white/10 text-white">
                <form onSubmit={onFormSubmit}>
                    <DialogHeader>
                        <DialogTitle>
                            {window.location.pathname === '/login' ? "Register New Account" : "Add New User"}
                        </DialogTitle>
                        <DialogDescription className="text-gray-400">
                            Enter details for the new user account.
                        </DialogDescription>
                    </DialogHeader>

                    <div className="grid gap-4 py-4">
                        <div className="grid gap-2">
                            <Label htmlFor="email">Email</Label>
                            <div className="relative">
                                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                                <Input
                                    id="email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    type="email"
                                    className="bg-white/5 border-white/10 pl-10" 
                                    placeholder="you@example.com"
                                    required 
                                />
                            </div>
                            
                            <Label htmlFor="password">Password</Label>
                            <div className="relative">
                                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                                <Input
                                    id="password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    type="password"
                                    className="bg-white/5 border-white/10 pl-10" 
                                    placeholder="*******"
                                    required 
                                    minLength={6}
                                />
                            </div>
                        </div>
                    </div>

                    <DialogFooter className="flex gap-2 justify-end">
                        <DialogClose asChild>
                            <Button type="button" variant="ghost" onClick={() => setIsOpen(false)}>
                                Cancel
                            </Button>
                        </DialogClose>
                        
                        <Button type="submit" className="bg-cyan-600 hover:bg-cyan-700">
                            {window.location.pathname === '/login' ? "Register" : "Add User"}
                        </Button>
                    </DialogFooter>
                </form>
            </DialogContent>
        </Dialog>
    )
}